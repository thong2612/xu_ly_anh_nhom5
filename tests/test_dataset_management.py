from __future__ import annotations

import csv
import io
import urllib.error
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.dataset import cli
from src.dataset.downloader import AuthenticationRequired, DownloadError, MendeleyDownloader
from src.dataset.inspector import METADATA_FIELDS, _component_metadata, _labels_from_path, _lineage, _mask_key, _split_from_path, inspect_datasets
from src.dataset.reporter import generate_report
from src.dataset.splitter import prepare_splits
from src.dataset.validator import validate_manifest
from src.utils.files import safe_relative_path


def test_ripeness_labels_require_exact_directory_component() -> None:
    labels = {"ripeness_labels": {
        "unripe": ["unripe"],
        "partially_ripe": ["partially ripe", "partially_ripe"],
        "ripe": ["ripe"],
    }}
    assert _labels_from_path(Path("source/Partially Ripe/fruit.jpg"), labels) == ("partially_ripe", "")
    assert _labels_from_path(Path("source/other/unripe_fruit.jpg"), labels) == ("", "")


def test_component_rules_are_exact_and_preserve_task_and_lineage() -> None:
    definition = {"component_rules": [
        {"path_component": "Damaged dataset", "task": "defect_analysis", "defect_label": "defective"},
        {"path_component": "BAD MANGOES AUGMENTED", "original_or_augmented": "augmented"},
    ]}
    path = Path("source/Damaged dataset/BAD MANGOES AUGMENTED/1.jpg")
    assert _component_metadata(path, definition) == {
        "task": "defect_analysis", "defect_label": "defective", "original_or_augmented": "augmented"
    }
    assert _component_metadata(Path("source/Damaged dataset copy/1.jpg"), definition) == {}


def test_mask_name_pair_key_removes_mask_suffix() -> None:
    assert _mask_key(Path("images/mango_01.jpg")) == _mask_key(Path("masks/mango_01_mask.png"))


def test_lineage_does_not_treat_the_raw_folder_as_original() -> None:
    assert _lineage(Path("data/raw/mango_ripeness/downloads/ripe/image.jpg")) == "unknown"
    assert _lineage(Path("data/raw/mango_ripeness/source/train_augmented/image.jpg")) == "augmented"


def test_published_split_is_detected_from_source_folders() -> None:
    assert _split_from_path(Path("data/raw/mango/source/train/class/image.jpg")) == ("train", "published")
    assert _split_from_path(Path("data/raw/mango/source/train/test/class/image.jpg")) == ("", "ambiguous")


def test_inspector_pairs_image_and_mask_but_keeps_semantics_unassigned(tmp_path: Path) -> None:
    source = tmp_path / "data" / "raw" / "mango_segmentation" / "source"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir(parents=True)
    Image.new("RGB", (8, 6), (220, 120, 40)).save(source / "images" / "fruit_01.png")
    Image.new("L", (8, 6), 255).save(source / "masks" / "fruit_01_mask.png")
    config = {
        "_project_root": str(tmp_path),
        "metadata_root": "data/metadata",
        "reports_root": "reports",
        "datasets": {"mango_segmentation": {
            "title": "segmentation source",
            "task": "defect_segmentation",
            "expected_images": 1,
            "ripeness_labels": {},
            "defect_labels": {},
        }},
    }
    inspect_datasets(config)
    manifest = tmp_path / "data" / "metadata" / "manifest.csv"
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["mask_path"].endswith("masks/fruit_01_mask.png")
    assert row["mask_type"] == ""
    assert row["validation_status"] == "mask_review_required"
    assert set(METADATA_FIELDS).issubset(row)

    with (tmp_path / "data" / "metadata" / "labels_overrides.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "ripeness_label", "defect_label", "mask_type", "mask_review_status", "original_or_augmented"])
        writer.writeheader()
        writer.writerow({"image_path": row["image_path"], "mask_type": "defect_region", "mask_review_status": "verified"})
    inspect_datasets(config)
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reviewed = next(csv.DictReader(stream))
    assert reviewed["mask_type"] == "defect_region"
    assert reviewed["mask_review_status"] == "verified"
    assert reviewed["validation_status"] == "valid"


def test_masks_do_not_pair_with_same_filename_from_another_component(tmp_path: Path) -> None:
    source = tmp_path / "data" / "raw" / "mango_segmentation" / "source"
    segmentation_images = source / "segmentation dataset" / "BAD MANGOES1"
    masks = source / "segmentation dataset" / "mango_mask"
    unrelated_images = source / "Damaged dataset" / "BAD MANGOES AUGMENTED"
    segmentation_images.mkdir(parents=True)
    masks.mkdir(parents=True)
    unrelated_images.mkdir(parents=True)
    Image.new("RGB", (8, 6), (220, 120, 40)).save(segmentation_images / "1.jpg")
    Image.new("L", (8, 6), 255).save(masks / "1.jpg")
    Image.new("RGB", (8, 6), (20, 180, 40)).save(unrelated_images / "1.jpg")
    config = {
        "_project_root": str(tmp_path),
        "metadata_root": "data/metadata",
        "reports_root": "reports",
        "datasets": {"mango_segmentation": {
            "title": "segmentation source",
            "task": "defect_segmentation",
            "expected_images": None,
            "ripeness_labels": {},
            "defect_labels": {},
            "component_rules": [
                {"path_component": "Damaged dataset", "task": "defect_analysis", "defect_label": "defective"},
                {"path_component": "segmentation dataset", "task": "defect_segmentation"},
            ],
        }},
    }

    summary = inspect_datasets(config)
    assert summary[0]["paired_mask_count"] == 1
    with (tmp_path / "data" / "metadata" / "manifest.csv").open("r", newline="", encoding="utf-8-sig") as stream:
        rows = {row["image_path"]: row for row in csv.DictReader(stream)}
    paired = next(row for row in rows.values() if row["task"] == "defect_segmentation")
    unrelated = next(row for row in rows.values() if row["task"] == "defect_analysis")
    assert paired["mask_path"].endswith("mango_mask/1.jpg")
    assert unrelated["mask_path"] == ""


def test_safe_relative_path_rejects_parent_traversal() -> None:
    assert safe_relative_path("class_a/image.jpg").as_posix() == "class_a/image.jpg"
    with pytest.raises(ValueError):
        safe_relative_path("../outside.txt")
    with pytest.raises(ValueError):
        safe_relative_path("C:/outside.txt")


def test_split_is_deterministic_and_keeps_identical_pixels_together(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    metadata = data_root / "metadata"
    metadata.mkdir(parents=True)
    source = data_root / "raw" / "mango_ripeness" / "source"
    source.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    pixel_hashes = [f"pixel-{i}" for i in range(10)] + ["pixel-0"]
    for index, pixel_hash in enumerate(pixel_hashes):
        image_path = source / f"fruit_{index}.png"
        pixel_value = 0 if pixel_hash == "pixel-0" else index % 255
        Image.new("RGB", (2, 2), (pixel_value, 0, 0)).save(image_path)
        rows.append({
            "image_id": f"id-{index}",
            "dataset": "mango_ripeness",
            "image_path": image_path.relative_to(tmp_path).as_posix(),
            "task": "ripeness_classification",
            "ripeness_label": "ripe",
            "defect_label": "",
            "mask_path": "",
            "mask_type": "",
            "split": "",
            "original_or_augmented": "original",
            "validation_status": "valid",
            "sha256": f"sha-{index}",
            "pixel_sha256": pixel_hash,
        })
    manifest = metadata / "manifest.csv"
    fieldnames = list(rows[0])
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "_project_root": str(tmp_path),
        "datasets": {"mango_ripeness": {"task": "ripeness_classification", "ripeness_labels": {"ripe": ["ripe"]}}},
        "split": {"seed": 42, "ratios": {"train": .7, "validation": .15, "test": .15}},
    }

    first = prepare_splits(config)
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        result_rows = list(csv.DictReader(stream))
    duplicate_splits = {row["split"] for row in result_rows if row["pixel_sha256"] == "pixel-0"}
    assert len(duplicate_splits) == 1
    assert first["prepared"][0]["counts"] == {"train": 7, "validation": 2, "test": 2}

    second = prepare_splits(config)
    assert second == first


def test_segmentation_split_skips_mask_size_mismatch_even_when_reviewed(tmp_path: Path) -> None:
    metadata = tmp_path / "data" / "metadata"
    metadata.mkdir(parents=True)
    row = {field: "" for field in METADATA_FIELDS}
    row.update({
        "image_id": "seg-1", "dataset": "mango_segmentation", "image_path": "data/raw/image.jpg",
        "task": "defect_segmentation", "mask_path": "data/raw/mask.jpg", "mask_type": "defect_region",
        "mask_review_status": "verified", "validation_status": "mask_size_mismatch",
        "sha256": "sha-1", "pixel_sha256": "pixels-1",
    })
    with (metadata / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    config = {
        "_project_root": str(tmp_path),
        "datasets": {"mango_segmentation": {"task": "defect_segmentation"}},
        "split": {"seed": 42, "ratios": {"train": .7, "validation": .15, "test": .15}},
    }

    summary = prepare_splits(config)
    assert summary["prepared"] == []
    assert summary["skipped"][0]["dataset"] == "mango_segmentation"
    assert "sai kích thước" in summary["skipped"][0]["reason"]


def test_official_split_is_preserved(tmp_path: Path) -> None:
    metadata = tmp_path / "data" / "metadata"
    metadata.mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        rows.append({
            "image_id": f"official-{index}", "dataset": "mango_ripeness", "image_path": f"data/raw/{split}/image_{index}.jpg",
            "task": "ripeness_classification", "ripeness_label": "unripe", "defect_label": "", "mask_path": "", "mask_type": "",
            "split": split, "split_origin": "published", "original_or_augmented": "original", "validation_status": "valid",
            "sha256": f"hash-{index}", "pixel_sha256": f"pixels-{index}",
        })
    manifest = metadata / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "_project_root": str(tmp_path),
        "datasets": {"mango_ripeness": {"task": "ripeness_classification", "ripeness_labels": {"unripe": ["unripe"]}}},
        "split": {"seed": 42, "ratios": {"train": .7, "validation": .15, "test": .15}},
    }
    summary = prepare_splits(config)
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        result = {row["image_id"]: row["split"] for row in csv.DictReader(stream)}
    assert result == {f"official-{index}": split for index, split in enumerate(("train", "validation", "test"))}
    assert summary["prepared"][0]["source_split_preserved"] is True


def test_download_stops_after_first_authentication_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    attempted: list[str] = []

    class FakeDownloader:
        def __init__(self, config: dict[str, object]):
            pass

        def download(self, keys: list[str]) -> dict[str, list[Path]]:
            attempted.extend(keys)
            raise AuthenticationRequired("HTTP 401")

    monkeypatch.setattr(cli, "MendeleyDownloader", FakeDownloader)
    assert cli._download({}, ["mango_ripeness", "mango_fruit_defect", "mango_segmentation"]) == 1
    assert attempted == ["mango_ripeness"]


def test_unauthorized_api_response_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    downloader = MendeleyDownloader({}, retries=4)

    class UnauthorizedOpener:
        calls = 0

        def open(self, request: object, timeout: int) -> None:
            self.calls += 1
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", None, None)

    opener = UnauthorizedOpener()
    monkeypatch.setattr(downloader, "opener", opener)
    with pytest.raises(AuthenticationRequired):
        downloader._open("https://api.data.mendeley.com/datasets/example/files")
    assert opener.calls == 1


def test_public_download_saves_only_valid_zip_and_skips_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dataset/readme.txt", "public zip")
    body = buffer.getvalue()

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/zip", "Content-Length": str(len(body))}

        def __init__(self) -> None:
            self.offset = 0

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def geturl(self) -> str:
            return "https://data.mendeley.com/public-api/zip/example/download/1"

        def read(self, size: int) -> bytes:
            result = body[self.offset:self.offset + size]
            self.offset += len(result)
            return result

    class FakeOpener:
        calls = 0

        def open(self, request: object, timeout: int) -> FakeResponse:
            self.calls += 1
            return FakeResponse()

    config = {
        "_project_root": str(tmp_path),
        "datasets": {"mango": {"dataset_id": "example", "version": 1}},
    }
    downloader = MendeleyDownloader(config)
    opener = FakeOpener()
    monkeypatch.setattr(downloader, "opener", opener)

    target = downloader.download_public_zips(["mango"])["mango"]
    assert zipfile.is_zipfile(target)
    assert target.stat().st_size == len(body)
    assert downloader.download_public_zips(["mango"])["mango"] == target
    assert opener.calls == 1


def test_public_download_resumes_with_matching_content_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dataset/readme.txt", "resume this archive")
    body = buffer.getvalue()
    offset = len(body) // 2
    partial = tmp_path / "data" / "raw" / "mango" / "downloads" / "mango_v1.zip.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(body[:offset])

    class FakeResponse:
        status = 206
        headers = {
            "Content-Type": "application/zip",
            "Content-Length": str(len(body) - offset),
            "Content-Range": f"bytes {offset}-{len(body) - 1}/{len(body)}",
        }

        def __init__(self) -> None:
            self.position = 0

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def geturl(self) -> str:
            return "https://data.mendeley.com/public-api/zip/example/download/1"

        def read(self, size: int) -> bytes:
            result = body[offset + self.position:offset + self.position + size]
            self.position += len(result)
            return result

    class FakeOpener:
        request = None

        def open(self, request: object, timeout: int) -> FakeResponse:
            self.request = request
            return FakeResponse()

    config = {"_project_root": str(tmp_path), "datasets": {"mango": {"dataset_id": "example", "version": 1}}}
    downloader = MendeleyDownloader(config)
    opener = FakeOpener()
    monkeypatch.setattr(downloader, "opener", opener)

    target = downloader.download_public_zips(["mango"])["mango"]
    assert opener.request.get_header("Range") == f"bytes={offset}-"
    assert zipfile.is_zipfile(target)


def test_public_download_rejects_html_and_stops_on_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class HtmlResponse:
        status = 200
        headers = {"Content-Type": "text/html"}

        def __enter__(self) -> "HtmlResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def geturl(self) -> str:
            return "https://data.mendeley.com/datasets/example/1"

        def read(self, size: int) -> bytes:
            return b"<html>sign in</html>"

    class HtmlOpener:
        def open(self, request: object, timeout: int) -> HtmlResponse:
            return HtmlResponse()

    config = {"_project_root": str(tmp_path), "datasets": {"mango": {"dataset_id": "example", "version": 1}}}
    downloader = MendeleyDownloader(config)
    monkeypatch.setattr(downloader, "opener", HtmlOpener())
    with pytest.raises(DownloadError, match="không trả về ZIP"):
        downloader.download_public_zips(["mango"])
    assert not list((tmp_path / "data" / "raw" / "mango" / "downloads").glob("*.part"))

    calls = 0

    class UnauthorizedOpener:
        def open(self, request: object, timeout: int) -> None:
            nonlocal calls
            calls += 1
            raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", None, None)

    monkeypatch.setattr(downloader, "opener", UnauthorizedOpener())
    with pytest.raises(AuthenticationRequired):
        downloader.download_public_zips(["mango"])
    assert calls == 1


def test_report_records_download_archive_status_and_size(tmp_path: Path) -> None:
    download_dir = tmp_path / "data" / "raw" / "mango" / "downloads"
    download_dir.mkdir(parents=True)
    archive_path = download_dir / "mango_v1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("readme.txt", "source archive")
    config = {
        "_project_root": str(tmp_path),
        "datasets": {"mango": {
            "title": "Mango source",
            "url": "https://data.mendeley.com/datasets/example/1",
            "task": "defect_analysis",
        }},
    }

    report = generate_report(config)
    download = report["datasets"][0]["download"]
    assert download["status"] == "downloaded"
    assert download["archives"] == [{
        "name": "mango_v1.zip",
        "size_bytes": archive_path.stat().st_size,
        "valid_zip": True,
    }]
    assert "mango_v1.zip" in (tmp_path / "reports" / "dataset_report.md").read_text(encoding="utf-8")


def test_empty_manifest_keeps_schema_through_split_and_validates(tmp_path: Path) -> None:
    metadata = tmp_path / "data" / "metadata"
    metadata.mkdir(parents=True)
    manifest = metadata / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        csv.DictWriter(stream, fieldnames=METADATA_FIELDS).writeheader()
    config = {
        "_project_root": str(tmp_path),
        "datasets": {"mango_ripeness": {"task": "ripeness_classification"}},
        "split": {"seed": 42, "ratios": {"train": .7, "validation": .15, "test": .15}},
    }

    prepare_splits(config)
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == METADATA_FIELDS
    assert validate_manifest(config) == {"rows": 0, "issue_count": 0, "issues": []}
