"""Dependency preflight; no installation or network access."""
import argparse
import importlib.util
import importlib
import json
from pathlib import Path
import sys


def check(backend):
    errors = []
    report = dict(backend=backend, python=sys.version.split()[0], errors=errors)
    modules = {'numpy':'numpy','cv2':'opencv-python','PIL':'Pillow','psutil':'psutil',
               'requests':'requests','win32gui':'pywin32','win32api':'pywin32','Crypto':'pycryptodome'}
    if backend.startswith('xfeat-'):
        modules.update(torch='torch', tqdm='tqdm')
    for module, package in modules.items():
        if importlib.util.find_spec(module) is None:
            errors.append(f'缺少依赖 {package}（模块 {module}）')
        elif module != 'torch':
            try:
                importlib.import_module(module)
            except Exception as exc:
                errors.append(f'依赖 {package} 无法载入：{type(exc).__name__}: {exc}')
    if backend.startswith('xfeat-'):
        weights = Path(__file__).resolve().parents[1]/'third_party/accelerated_features/weights/xfeat.pt'
        report['weights_found'] = weights.is_file()
        if not weights.is_file():
            errors.append('缺少模型：no_minimap_lab/third_party/xfeat/weights/xfeat.pt')
        if importlib.util.find_spec('torch') is not None:
            try:
                import torch
                report.update(torch=torch.__version__, torch_cuda=torch.version.cuda)
                if backend == 'xfeat-cuda':
                    if torch.version.cuda is None:
                        errors.append('当前 PyTorch 是 CPU 版，不能使用 XFeat GPU；需安装支持 CUDA 的 PyTorch，或选择 XFeat CPU / SIFT CPU')
                    elif not torch.cuda.is_available():
                        errors.append('PyTorch 含 CUDA，但 GPU 不可用；请检查 NVIDIA 显卡和驱动，或选择 CPU 后端')
                    else:
                        # Exercise the runtime instead of trusting availability alone.
                        torch.zeros(8,device='cuda').sum().item()
                        report['gpu'] = torch.cuda.get_device_name(0)
            except Exception as exc:
                errors.append(f'PyTorch / CUDA 运行检查失败：{type(exc).__name__}: {exc}')
    report['ok'] = not errors
    return report


def require(backend):
    report = check(backend)
    if not report['ok']:
        raise RuntimeError('运行环境检查未通过：\n'+'\n'.join(report['errors']))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('sift-cpu','xfeat-cpu','xfeat-cuda'),default='xfeat-cuda')
    report=check(parser.parse_args().backend)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['ok'] else 1


if __name__=='__main__':
    raise SystemExit(main())
