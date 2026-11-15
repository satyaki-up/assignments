import argparse
import hashlib
import json
import os
import urllib.parse
from pathlib import Path

from hf_transfer import download as hf_transfer_download

from data import (
    DEFAULT_DATA_REPO_ID,
    DEFAULT_DATA_REVISION,
    MANIFEST_FILE,
    default_dataset_root,
)


CHUNK_BYTES = 8 * 1024 * 1024
HF_TRANSFER_CHUNK_BYTES = 10 * 1024 * 1024
HF_TRANSFER_MAX_FILES = 100
HF_TRANSFER_MAX_RETRIES = 5
HF_TRANSFER_PARALLEL_FAILURES = 3
PROGRESS_EVERY_BYTES = 512 * 1024 * 1024


def hf_file_url(repo_id, revision, filename):
    quoted_file = urllib.parse.quote(filename)
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_file}"


def _format_bytes(num_bytes):
    num_bytes = int(num_bytes)
    for scale, suffix in [
        (1024**3, "GiB"),
        (1024**2, "MiB"),
        (1024, "KiB"),
    ]:
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {suffix}"
    return f"{num_bytes} bytes"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_file_is_valid(path, expected_size=None, expected_sha256=None):
    path = Path(path)
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        return False
    return True


def _hf_headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def _download_file(url, path, *, expected_size=None, expected_sha256=None, force=False):
    path = Path(path)
    if not force and _existing_file_is_valid(path, expected_size, expected_sha256):
        print(f"Already present: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"Downloading {url} -> {path}")
    bytes_reported = 0
    next_report = PROGRESS_EVERY_BYTES

    def _on_chunk(num_bytes):
        nonlocal bytes_reported, next_report
        bytes_reported += int(num_bytes)
        if expected_size and bytes_reported >= next_report:
            print(
                f"  {_format_bytes(bytes_reported)} / "
                f"{_format_bytes(expected_size)}"
            )
            next_report += PROGRESS_EVERY_BYTES

    try:
        hf_transfer_download(
            url=url,
            filename=str(tmp_path),
            max_files=HF_TRANSFER_MAX_FILES,
            chunk_size=HF_TRANSFER_CHUNK_BYTES,
            parallel_failures=HF_TRANSFER_PARALLEL_FAILURES,
            max_retries=HF_TRANSFER_MAX_RETRIES,
            headers=_hf_headers(),
            callback=_on_chunk,
        )
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    if not tmp_path.exists():
        raise RuntimeError(f"Failed to download {url}: no file was written.")

    actual_size = tmp_path.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        tmp_path.unlink()
        raise RuntimeError(
            f"Downloaded {actual_size} bytes for {path}, expected {expected_size}."
        )
    actual_sha256 = _file_sha256(tmp_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        tmp_path.unlink()
        raise RuntimeError(
            f"sha256 mismatch for {path}: got {actual_sha256}, "
            f"expected {expected_sha256}."
        )

    os.replace(tmp_path, path)


def download_default_data(
    output_dir=None,
    *,
    repo_id=DEFAULT_DATA_REPO_ID,
    revision=DEFAULT_DATA_REVISION,
    force=False,
):
    output_dir = Path(output_dir) if output_dir is not None else default_dataset_root()
    manifest_path = output_dir / MANIFEST_FILE
    manifest_url = hf_file_url(repo_id, revision, MANIFEST_FILE)
    _download_file(manifest_url, manifest_path, force=force)

    manifest = json.loads(manifest_path.read_text())
    files = manifest.get("files", [])
    if not files:
        raise ValueError(f"{manifest_path} does not list any files.")

    for file_info in files:
        filename = file_info["path"]
        _download_file(
            hf_file_url(repo_id, revision, filename),
            output_dir / filename,
            expected_size=file_info.get("size_bytes"),
            expected_sha256=file_info.get("sha256"),
            force=force,
        )

    print(f"Data ready at {output_dir}")
    return {
        "data_dir": str(output_dir),
        "repo_id": repo_id,
        "revision": revision,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description="Download Deep Learning Alchemy token data.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repo-id", default=DEFAULT_DATA_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_DATA_REVISION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_default_data(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        revision=args.revision,
        force=args.force,
    )


if __name__ == "__main__":
    main()
