"""Reuse main calibration schema in a lab-owned template directory."""
import shutil
from .atlas import ROOT

TEMPLATES = ROOT / 'no_minimap_lab' / 'calibration' / 'templates'


def calibration_detector():
    from src.vision.main_view_detector import MainViewDetector
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / 'assets' / 'templates').glob('player*'):
        target = TEMPLATES / source.name
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
    return MainViewDetector(template_dir=str(TEMPLATES),
                            monster_compute_device='cpu', monster_hp_bar_compute_device='cpu')
