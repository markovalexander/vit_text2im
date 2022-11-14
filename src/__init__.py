from pathlib import Path


def root_path() -> Path:
    return Path(__file__).parent.parent.absolute()

