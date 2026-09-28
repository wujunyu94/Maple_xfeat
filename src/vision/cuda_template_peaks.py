"""Optional CUDA coarse NCC. Fine validation always uses the original OpenCV score."""
from __future__ import annotations

import numpy as np


class CudaTemplatePeaks:
    def __init__(self):
        # Keep PyTorch optional for CPU-only installations.
        import torch
        import torch.nn.functional as functional
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        self.torch = torch
        self.functional = functional
        self._key = None
        self._weights = None
        self._entries = []

    def prepare(self, key, templates, prepared):
        if self._key == key:
            return
        entries = []
        for tpl in templates:
            for index, entry in enumerate(prepared[id(tpl)]):
                gray, mask, _ = entry[5]
                # Masked CCORR has different score semantics; use the CPU path.
                if mask is not None:
                    continue
                centered = gray.astype(np.float32) / 255.0
                centered -= centered.mean()
                norm = float(np.linalg.norm(centered))
                if norm < 1e-6:
                    continue
                entries.append(((id(tpl), index), centered / norm))
        self._entries = entries
        self._weights = None
        if entries:
            self._height = max(g.shape[0] for _, g in entries)
            self._width = max(g.shape[1] for _, g in entries)
            weights = np.zeros((len(entries), 1, self._height, self._width), np.float32)
            for i, (_, gray) in enumerate(entries):
                weights[i, 0, :gray.shape[0], :gray.shape[1]] = gray
            self._weights = self.torch.from_numpy(weights).to('cuda')
        self._key = key

    def prepare_frame(self, gray):
        """Upload one coarse frame and build integrals shared by mob matchers."""
        torch, F = self.torch, self.functional
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(gray)).to('cuda').float()[None, None] / 255.0
            # Double precision is required on uniform backgrounds; sharing
            # these tensors must not change the existing NCC score semantics.
            plane = x[0, 0].double()
            sums = F.pad(plane.cumsum(0).cumsum(1), (1, 0, 1, 0))
            squares = F.pad(plane.square().cumsum(0).cumsum(1), (1, 0, 1, 0))
        return x, sums, squares

    def match(self, gray, threshold, limit, *, frame_context=None):
        """Return only bounded peak lists; correlation maps stay on the GPU."""
        if self._weights is None:
            return {}
        torch, F = self.torch, self.functional
        height, width = gray.shape
        valid = [(i, key, g.shape) for i, (key, g) in enumerate(self._entries)
                 if g.shape[0] < height and g.shape[1] < width]
        if not valid:
            return {}
        with torch.inference_mode():
            x, sums, squares = (
                self.prepare_frame(gray) if frame_context is None else frame_context
            )
            if x.shape[-2:] != (height, width):
                raise ValueError("shared CUDA frame shape does not match coarse image")
            # Disable TF32 for NCC: rounding small eye features changes near-threshold peaks.
            with torch.backends.cudnn.flags(enabled=True, benchmark=False, allow_tf32=False):
                corr = F.conv2d(F.pad(x, (0, self._width - 1, 0, self._height - 1)), self._weights)[0]
            packed = []
            for i, key, (h, w) in valid:
                total = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
                total_sq = squares[h:, w:] - squares[:-h, w:] - squares[h:, :-w] + squares[:-h, :-w]
                variance = (total_sq - total.square() / (h * w)).clamp_min(0).float()
                response = corr[i, :total.shape[0], :total.shape[1]] / variance.clamp_min(1e-12).sqrt()
                response = torch.where(variance > 1e-10, response.clamp(-1, 1), 0.0)
                peaks = F.max_pool2d(response[None, None], 5, 1, 2)[0, 0]
                response = torch.where((response >= peaks) & (response >= threshold), response, -torch.inf)
                count = min(limit, response.numel())
                scores, indices = torch.topk(response.flatten(), count)
                values = torch.stack((scores, indices // response.shape[1], indices % response.shape[1]))
                packed.append(F.pad(values, (0, limit - count), value=-float('inf')))
            # One small device-to-host transfer for all templates and scales.
            values = torch.stack(packed).cpu().numpy()
        result = {}
        for (_, key, _), row in zip(valid, values):
            keep = np.isfinite(row[0])
            scores, ys, xs = row[:, keep]
            # CPU np.where emits row order when the candidate limit is not reached.
            if len(scores) < limit:
                order = np.lexsort((xs, ys))
                scores, ys, xs = scores[order], ys[order], xs[order]
            result[key] = (ys.astype(np.int32), xs.astype(np.int32), scores)
        return result
