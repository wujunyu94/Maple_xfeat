"""Validate that the pinned XFeat Git submodule is initialized."""
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "third_party" / "accelerated_features"
    required = (root / "modules" / "xfeat.py", root / "weights" / "xfeat.pt", root / "LICENSE")
    missing = [str(path.relative_to(root.parent.parent)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("XFeat submodule is incomplete. Run `git submodule update --init --recursive`. Missing: " + ", ".join(missing))
    print("XFeat submodule and checkpoint are ready.")


if __name__ == "__main__":
    main()
