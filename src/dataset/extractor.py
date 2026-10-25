from __future__ import annotations

import json
import logging
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from src.utils.config import project_path
from src.utils.files import safe_relative_path, sha256_file, slugify


LOGGER = logging.getLogger(__name__)


def extract_archives(config: dict[str, Any], dataset_keys: list[str] | None = None) -> dict[str, int]:
    keys = dataset_keys or list(config["datasets"])
    result: dict[str, int] = {}
    for key in keys:
        downloads = project_path(config, f"data/raw/{key}/downloads")
        source = project_path(config, f"data/raw/{key}/source")
        count = 0
        processed: set[Path] = set()
        while True:
            roots = [path for path in (downloads, source) if path.exists()]
            candidates = sorted(
                {
                    archive
                    for root in roots
                    for archive in root.rglob("*")
                    if archive.is_file() and archive.suffix.casefold() in {".zip", ".rar"}
                },
                key=lambda path: str(path).casefold(),
            )
            archive = next((path for path in candidates if path not in processed), None)
            if archive is None:
                break
            processed.add(archive)
            try:
                is_rar = archive.suffix.casefold() == ".rar"
                if not is_rar and not zipfile.is_zipfile(archive):
                    LOGGER.warning("File có đuôi ZIP nhưng không hợp lệ: %s", archive)
                    continue
                digest = sha256_file(archive)
                nested_archive = source in archive.parents
                destination = (
                    archive.parent / f"{slugify(archive.stem)}_extracted"
                    if is_rar and nested_archive
                    else source / (f"{slugify(archive.stem)}_extracted" if is_rar else slugify(archive.stem))
                )
                marker = destination / ".mangoresearch_extract.json"
                if marker.exists():
                    old = json.loads(marker.read_text(encoding="utf-8"))
                    if old.get("archive_sha256") == digest:
                        LOGGER.info("Đã giải nén bản này, bỏ qua: %s", archive.name)
                        continue
                if destination.exists():
                    destination = source / f"{slugify(archive.stem)}_{digest[:10]}"
                    marker = destination / ".mangoresearch_extract.json"
                    if marker.exists():
                        old = json.loads(marker.read_text(encoding="utf-8"))
                        if old.get("archive_sha256") == digest:
                            continue
                    if destination.exists():
                        LOGGER.warning("Không ghi đè thư mục đã tồn tại: %s", destination)
                        continue
                destination.mkdir(parents=True, exist_ok=False)
                if is_rar:
                    tar = shutil.which("tar")
                    if not tar:
                        raise RuntimeError("Không tìm thấy tar/bsdtar để giải nén RAR; cài 7-Zip hoặc bsdtar.")
                    listing = subprocess.run(
                        [tar, "-tf", str(archive)], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", check=True,
                    )
                    members = [line.rstrip("\r") for line in listing.stdout.splitlines() if line.rstrip("\r")]
                    for member in members:
                        relative = safe_relative_path(member)
                        if not relative.parts:
                            raise ValueError(f"Đường dẫn rỗng trong RAR: {archive}")
                    verbose = subprocess.run(
                        [tar, "-tvf", str(archive)], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", check=True,
                    )
                    type_lines = [line for line in verbose.stdout.splitlines() if line.strip()]
                    if len(type_lines) != len(members) or any(line[0] not in {"-", "d"} for line in type_lines):
                        raise ValueError(f"RAR có kiểu thành viên không hỗ trợ hoặc listing không nhất quán: {archive}")
                    subprocess.run(
                        [tar, "-xf", str(archive), "-C", str(destination), "--no-same-owner", "--no-same-permissions"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
                    )
                else:
                    with zipfile.ZipFile(archive) as bundle:
                        for member in bundle.infolist():
                            relative = safe_relative_path(member.filename)
                            mode = member.external_attr >> 16
                            if stat.S_ISLNK(mode):
                                LOGGER.warning("Bỏ qua symlink trong archive: %s", member.filename)
                                continue
                            target = destination.joinpath(*relative.parts)
                            if member.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.open(member) as src, target.open("wb") as dst:
                                while chunk := src.read(1024 * 1024):
                                    dst.write(chunk)
                marker.write_text(
                    json.dumps({"archive_sha256": digest, "archive_name": archive.name}, indent=2),
                    encoding="utf-8",
                )
                count += 1
                LOGGER.info("Đã giải nén %s vào %s", archive.name, destination)
            except (OSError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile, ValueError, json.JSONDecodeError) as exc:
                LOGGER.error("Không giải nén được %s: %s", archive, exc)
        result[key] = count
    return result
