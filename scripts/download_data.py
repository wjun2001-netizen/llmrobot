"""Download only official text games and required trajectory metadata, with TLS verification."""
import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Digests recorded from the official HTTPS release assets during initial setup.
ASSETS = (
    ("0.2.2", "json_2.1.1_json.zip", "25171f16e20ad7b048c47275c45b0babf3aa1cbab29cec97387922350a9844bc"),
    ("0.4.0", "json_2.1.2_tw-pddl.zip", "eea90499df27b9cb3147d1d1927264146d38194b575e25f70a82875545742dcb"),
)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(destination):
    destination = Path(destination).resolve()
    downloads = destination / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    manifest = []
    for version, name, digest in ASSETS:
        url = f"https://github.com/alfworld/alfworld/releases/download/{version}/{name}"
        archive = downloads / name
        if not archive.exists():
            temporary = downloads / (name + ".part")
            # A failed download is retained for diagnosis; never treat it as complete.
            if temporary.exists():
                raise FileExistsError(f"Inspect incomplete download before retrying: {temporary}")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if sha256(temporary) != digest:
                raise ValueError(f"SHA256 mismatch: {temporary}; do not use this archive")
            os.rename(temporary, archive)
        if sha256(archive) != digest:
            raise ValueError(f"SHA256 mismatch: {archive}")
        with zipfile.ZipFile(archive) as zipped:
            if zipped.testzip() is not None:
                raise ValueError(f"ZIP CRC failure: {archive}")
            for info in zipped.infolist():
                target = (destination / info.filename).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError(f"Unsafe archive member: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                content = zipped.read(info)
                if target.exists():
                    if target.read_bytes() != content:
                        raise ValueError(f"Existing data differs; refusing overwrite: {target}")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(content)
        manifest.append({"url": url, "sha256": digest, "archive": name})
        print(f"Verified and extracted: {name}", flush=True)
    manifest_path = destination / "dataset_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing dataset manifest differs")
    else:
        with manifest_path.open("x") as stream:
            json.dump(manifest, stream, indent=2)
    print(f"Ready: {destination / 'json_2.1.1'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/alfworld")
    prepare(parser.parse_args().data_dir)
