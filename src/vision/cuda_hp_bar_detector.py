"""PyTorch CUDA implementation of the classic-game monster HP-bar detector.

Only the final, filtered bar rectangles cross back to the CPU.  The existing
OpenCV implementation remains the reference implementation and fallback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


BarRect = Tuple[int, int, int, int]


class CudaHpBarDetector:
    """Vectorised full/partially-occluded HP-bar detection on CUDA."""

    def __init__(self) -> None:
        import torch
        import torch.nn.functional as functional

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is unavailable")
        self.torch = torch
        self.functional = functional
        self.device_name = str(torch.cuda.get_device_name(0))

    @staticmethod
    def _deduplicate(candidates: Sequence[BarRect]) -> List[BarRect]:
        final: List[BarRect] = []
        for candidate in candidates:
            cx, cy, _cw, _ch = candidate
            if not any(
                abs(cx - fx) <= 15 and abs(cy - fy) <= 3
                for fx, fy, _fw, _fh in final
            ):
                final.append(candidate)
        return final

    def detect(
        self,
        bgr_frame: np.ndarray,
        exclusion_regions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[BarRect]:
        torch = self.torch
        F = self.functional
        if bgr_frame is None or bgr_frame.size == 0 or bgr_frame.ndim != 3:
            return []
        height, width = bgr_frame.shape[:2]
        if height < 20 or width < 60:
            return []

        with torch.inference_mode():
            # Capture frames are normally contiguous, but ROI/debug callers are
            # allowed to pass a view.  Make contiguity explicit before H2D.
            host = np.ascontiguousarray(bgr_frame)
            bgr = torch.from_numpy(host).to("cuda", non_blocking=False)
            white = ((bgr >= 230) & (bgr <= 255)).all(dim=2)

            if height >= 300:
                white[max(0, height - 45):height, :] = False
            for exclusion in exclusion_regions or ():
                try:
                    if not exclusion.get("monster", True):
                        continue
                    ex_x = max(0, int(exclusion.get("x", 0)))
                    ex_y = max(0, int(exclusion.get("y", 0)))
                    ex_w = max(0, int(exclusion.get("w", 0)))
                    ex_h = max(0, int(exclusion.get("h", 0)))
                except Exception:
                    continue
                if ex_w > 0 and ex_h > 0:
                    white[
                        ex_y:min(height, ex_y + ex_h),
                        ex_x:min(width, ex_x + ex_w),
                    ] = False

            # Valid erosions only: all subsequent anchors use top-left positions
            # whose complete 14/50/8-pixel kernel lies inside the frame.
            white_f = white.float()[None, None]
            h14 = F.avg_pool2d(white_f, (1, 14), stride=1)[0, 0] == 1.0
            h50 = F.avg_pool2d(white_f, (1, 50), stride=1)[0, 0] == 1.0
            v8 = F.avg_pool2d(white_f, (8, 1), stride=1)[0, 0] == 1.0

            blue = bgr[:, :, 0]
            green = bgr[:, :, 1]
            red = bgr[:, :, 2]
            coloured = (
                ((green > 160) & (red < 90) & (blue < 90))
                | ((red > 130) & (green < 90) & (blue < 90))
                | ((red < 50) & (green < 50) & (blue < 50))
            )
            colour_integral = F.pad(
                coloured.to(torch.int32).cumsum(0).cumsum(1),
                (1, 0, 1, 0),
            )

            def colour_count(x1, y1, x2, y2):
                return (
                    colour_integral[y2, x2]
                    - colour_integral[y1, x2]
                    - colour_integral[y2, x1]
                    + colour_integral[y1, x1]
                )

            # Row prefix of B+G+R preserves the CPU mean<=65 black-border test
            # without copying 50-pixel strips back to Python.
            intensity = bgr.to(torch.int32).sum(dim=2)
            row_prefix = F.pad(intensity.cumsum(1), (1, 0))

            full_anchor = (
                h50[:-7, :]
                & h50[7:, :]
                & v8[:, :width - 49]
                & v8[:, 49:]
                & ~white[3:height - 4, 25:width - 24]
            )
            full_coords = torch.nonzero(full_anchor, as_tuple=False)
            if full_coords.numel():
                fy_all, fx_all = full_coords[:, 0], full_coords[:, 1]
                keep = torch.ones_like(fx_all, dtype=torch.bool)
                has_top = fy_all >= 1
                top_y = (fy_all - 1).clamp_min(0)
                top_sum = row_prefix[top_y, fx_all + 50] - row_prefix[top_y, fx_all]
                keep &= (~has_top) | (top_sum <= 65 * 50 * 3)
                has_bottom = fy_all + 8 < height
                bottom_y = (fy_all + 8).clamp_max(height - 1)
                bottom_sum = (
                    row_prefix[bottom_y, fx_all + 50]
                    - row_prefix[bottom_y, fx_all]
                )
                keep &= (~has_bottom) | (bottom_sum <= 65 * 50 * 3)
                inner = colour_count(
                    fx_all + 2, fy_all + 2, fx_all + 48, fy_all + 5
                )
                keep &= inner * 100 >= (3 * 46) * 70
                full_y = fy_all[keep]
                full_x = fx_all[keep]
            else:
                full_y = torch.empty(0, dtype=torch.long, device="cuda")
                full_x = torch.empty(0, dtype=torch.long, device="cuda")

            pair_rows = white[:-7, :] & white[7:, :]

            def reject_near_full(y, x, *, right_edge: bool = False):
                if full_x.numel() == 0 or x.numel() == 0:
                    return torch.ones_like(x, dtype=torch.bool)
                if right_edge:
                    near_edge = (
                        (y[:, None] - full_y[None, :]).abs() <= 3
                    ) & (
                        (x[:, None] - (full_x[None, :] + 49)).abs() <= 15
                    )
                    inferred = x - 49
                else:
                    near_edge = (
                        (y[:, None] - full_y[None, :]).abs() <= 3
                    ) & (
                        (x[:, None] - full_x[None, :]).abs() <= 15
                    )
                    inferred = x
                near_origin = (
                    (y[:, None] - full_y[None, :]).abs() <= 3
                ) & (
                    (inferred[:, None] - full_x[None, :]).abs() <= 15
                )
                return ~(near_edge | near_origin).any(dim=1)

            offsets = torch.arange(48, device="cuda", dtype=torch.long)
            empty_coord = torch.empty(0, dtype=torch.long, device="cuda")

            # Left-visible partial bars.
            left_anchor = (
                h14[:-7, :]
                & h14[7:, :]
                & v8[:, :width - 13]
                & ~white[3:height - 4, 7:width - 6]
            )
            if full_anchor.numel():
                left_anchor[:, :width - 49] &= ~full_anchor
            left_coords = torch.nonzero(left_anchor, as_tuple=False)
            left_x = empty_coord
            left_y = empty_coord
            if left_coords.numel():
                ly, lx = left_coords[:, 0], left_coords[:, 1]
                keep = reject_near_full(ly, lx)
                ly, lx = ly[keep], lx[keep]
                indices = lx[:, None] + offsets[None, :]
                in_bounds = indices < width
                values = pair_rows[
                    ly[:, None], indices.clamp_max(width - 1)
                ] & in_bounds
                stopped = ~values
                has_stop = stopped.any(dim=1)
                run = torch.where(
                    has_stop,
                    stopped.to(torch.int8).argmax(dim=1),
                    torch.full_like(lx, 48),
                )
                right = lx + run - 1
                inferred = lx
                valid = inferred + 49 < width
                vis_x1 = inferred + 2
                vis_x2 = torch.minimum(right, inferred + 47)
                visible_width = vis_x2 - vis_x1 + 1
                valid &= (vis_x2 - vis_x1) >= 10
                coloured_pixels = colour_count(
                    vis_x1, ly + 2, vis_x2 + 1, ly + 5
                )
                valid &= coloured_pixels * 100 >= visible_width * 3 * 70
                left_x = inferred[valid]
                left_y = ly[valid]

            # Right-visible partial bars.  The 14-pixel horizontal anchor starts
            # 13 pixels before the right vertical edge, matching the CPU layout.
            right_base = (
                h14[:-7, :]
                & h14[7:, :]
                & v8[:, 13:]
                & ~white[3:height - 4, 7:width - 6]
            )
            right_coords = torch.nonzero(right_base, as_tuple=False)
            right_x = empty_coord
            right_y = empty_coord
            if right_coords.numel():
                ry = right_coords[:, 0]
                rx = right_coords[:, 1] + 13
                keep = reject_near_full(ry, rx, right_edge=True)
                ry, rx = ry[keep], rx[keep]
                indices = rx[:, None] - offsets[None, :]
                in_bounds = indices >= 0
                values = pair_rows[
                    ry[:, None], indices.clamp_min(0)
                ] & in_bounds
                stopped = ~values
                has_stop = stopped.any(dim=1)
                run = torch.where(
                    has_stop,
                    stopped.to(torch.int8).argmax(dim=1),
                    torch.full_like(rx, 48),
                )
                left = rx - run + 1
                inferred = rx - 49
                valid = (inferred >= 0) & (inferred + 49 < width)
                vis_x1 = torch.maximum(left, inferred + 2)
                vis_x2 = torch.minimum(rx, inferred + 47)
                visible_width = vis_x2 - vis_x1 + 1
                valid &= (vis_x2 - vis_x1) >= 10
                coloured_pixels = colour_count(
                    vis_x1, ry + 2, vis_x2 + 1, ry + 5
                )
                valid &= coloured_pixels * 100 >= visible_width * 3 * 70
                right_x = inferred[valid]
                right_y = ry[valid]

            # Exactly one device-to-host synchronization per frame.  Separate
            # full/left/right .cpu() calls cost milliseconds even for empty
            # result sets, while only a handful of rectangles are returned.
            all_x = torch.cat((full_x, left_x, right_x))
            all_y = torch.cat((full_y, left_y, right_y))
            packed = torch.stack((all_x, all_y), dim=1).cpu().tolist()
            results = [
                (max(0, int(x) - 1), max(0, int(y) - 1), 52, 10)
                for x, y in packed
            ]
            return self._deduplicate(results)
