"""Download and verify the two training inputs attached to the project release."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
RELEASE = (
    "https://github.com/ZiyaoYang-roseate/CLhypformer/"
    "releases/download/training-inputs/"
)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name, metadata, destination):
    with TemporaryDirectory(prefix=".training-inputs-", dir=ROOT) as temporary:
        staged = Path(temporary) / name
        for attempt in range(3):
            try:
                with urlopen(RELEASE + name, timeout=90) as response, staged.open("wb") as file:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        file.write(block)
                if staged.stat().st_size != metadata["bytes"] or checksum(staged) != metadata["sha256"]:
                    raise ValueError(f"Downloaded input failed verification: {name}")
                staged.replace(destination)
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                sleep(2 ** (attempt + 1))


def main():
    manifest = json.loads((ROOT / "training_input_assets.json").read_text(encoding="utf-8"))
    for name, metadata in manifest.items():
        destination = ROOT / name
        if (destination.exists() and destination.stat().st_size == metadata["bytes"]
                and checksum(destination) == metadata["sha256"]):
            continue
        download(name, metadata, destination)


if __name__ == "__main__":
    main()
