"""Download and verify the input archives attached to the GitHub release."""

from concurrent.futures import ThreadPoolExecutor
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


def download_part(part, staging):
    path = staging / part["name"]
    for attempt in range(3):
        try:
            with urlopen(RELEASE + part["name"], timeout=90) as response, path.open("wb") as file:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    file.write(block)
            if path.stat().st_size != part["bytes"] or checksum(path) != part["sha256"]:
                raise ValueError(f"Incomplete or corrupt download: {part['name']}")
            return path
        except (OSError, ValueError):
            if attempt == 2:
                raise
            sleep(2 ** (attempt + 1))


def main():
    manifest = json.loads((ROOT / "training_input_assets.json").read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        destination = ROOT / name
        if destination.exists() and checksum(destination) == entry["sha256"]:
            continue
        with TemporaryDirectory(prefix=".training-inputs-", dir=ROOT) as temporary:
            staging = Path(temporary)
            with ThreadPoolExecutor(max_workers=4) as threads:
                parts = list(threads.map(lambda part: download_part(part, staging), entry["parts"]))
            joined = staging / name
            with joined.open("wb") as output:
                for path in parts:
                    with path.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            output.write(block)
                    path.unlink()
            if checksum(joined) != entry["sha256"]:
                raise ValueError(f"Input checksum differs: {name}")
            joined.replace(destination)


if __name__ == "__main__":
    main()
