from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.dataset.downloader import AuthenticationRequired, DownloadError, MendeleyDownloader
from src.dataset.extractor import extract_archives
from src.dataset.inspector import inspect_datasets
from src.dataset.organizer import organize_images
from src.dataset.phase02 import run_phase02
from src.dataset.reporter import generate_report
from src.dataset.splitter import prepare_splits
from src.dataset.validator import validate_manifest
from src.color.pipeline import run_phase1
from src.segmentation.pipeline import approve_fruit_masks, import_fruit_masks, run_phase2
from src.segmentation.annotation_gui import annotate_fruit_masks
from src.segmentation.phase3 import run_phase3
from src.features.phase4 import run_phase4
from src.utils.config import CONFIG_PATH, load_config
from src.utils.config import project_path


LOGGER = logging.getLogger("mangoresearch")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MangoResearch Phase 0 dataset manager")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Đường dẫn config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("download", "Tải file qua API chính thức Mendeley"),
        ("download-public", "Tải ZIP Download All công khai, hỗ trợ resume"),
        ("extract", "Giải nén ZIP/RAR an toàn vào vùng source riêng"),
        ("inspect", "Kiểm tra ảnh/mask và tạo manifest, preview"),
        ("validate", "Kiểm tra lại đường dẫn, ảnh/mask và rò rỉ split"),
        ("organize", "Sao chép ảnh theo task/nhãn đã xác minh"),
        ("split", "Tạo split 70/15/15 khi đủ điều kiện"),
        ("pipeline", "Chạy toàn bộ các bước Phase 0"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--dataset", action="append", help="Key dataset; lặp lại để chọn nhiều nguồn")
    subparsers.add_parser("report", help="Xuất báo cáo Markdown/JSON")
    subparsers.add_parser("phase02", help="Audit trùng lặp/lineage, chuẩn hóa mask và tạo split provisional")
    phase1 = subparsers.add_parser("phase1", help="So sánh Gray World và thống kê màu HSV/CIELAB trên mẫu sạch")
    phase1.add_argument("--dataset", action="append", help="Key dataset; lặp lại để chọn nhiều nguồn")
    phase1.add_argument("--samples-per-group", type=int, help="Số ảnh cho mỗi nhóm dataset/task/nhãn")
    phase1.add_argument("--no-gray-world", action="store_true", help="Chỉ chạy cấu hình gốc; cache tách biệt")
    phase1.add_argument("--force", action="store_true", help="Bỏ qua cache và tính lại mẫu đã chọn")
    phase2 = subparsers.add_parser("phase2-segmentation", help="Chạy baseline phân đoạn xoài một kênh trên mẫu sạch")
    phase2.add_argument("--dataset", action="append", help="Key dataset; lặp lại để chọn nhiều nguồn")
    phase2.add_argument("--samples-per-group", type=int, help="Số ảnh mỗi nhóm dataset/task/nhãn")
    phase2.add_argument("--limit", type=int, help="Giới hạn tổng số mẫu sau khi chọn xác định")
    phase2.add_argument("--image-id", help="Chọn một image_id thuộc mẫu sạch được chọn")
    phase2.add_argument("--no-gray-world", action="store_true", help="Chỉ chạy E1; cache được phân biệt")
    phase2.add_argument("--force", action="store_true", help="Tính lại dù có cache")
    annotate = subparsers.add_parser("annotate-fruit-masks", help="Mở GUI polygon để tạo mask manual của toàn quả")
    annotate.add_argument("--limit", type=int, help="Giới hạn số candidate đầu vào")
    importer = subparsers.add_parser("import-fruit-masks", help="Import mask 0/255 manual đã lưu theo image_id.png")
    importer.add_argument("--input-dir", required=True, help="Thư mục PNG mask manual")
    approver = subparsers.add_parser("approve-fruit-masks", help="Xác nhận semantic thủ công cho mask đã import và kiểm tra định dạng")
    approver.add_argument("--image-id", action="append", required=True,
                          help="ID đã được người chú giải kiểm tra là mask toàn quả; lặp lại cho nhiều ảnh")
    phase3 = subparsers.add_parser("phase3-kmeans", help="Chạy K-Means HSV/Lab trên fruit mask và đánh giá exploratory với Dataset C")
    phase3.add_argument("--limit", type=int, help="Giới hạn số cặp Dataset C đầu tiên; mặc định chạy toàn bộ cặp đủ điều kiện")
    phase3.add_argument("--mode", action="append", choices=["heuristic", "approved"],
                        help="Chế độ mask; lặp lại để chọn cả hai, mặc định theo config")
    phase3.add_argument("--force", action="store_true", help="Bỏ qua cache K-Means và tính lại")
    phase4 = subparsers.add_parser("phase4-features", help="Trích xuất feature HSV/Lab và K-Means cho các split Phase 0.2")
    phase4.add_argument("--task", action="append", choices=["ripeness", "defect"],
                        help="Chọn nhánh; mặc định chạy cả ripeness và defect")
    phase4.add_argument("--limit", type=int, help="Giới hạn số ảnh mỗi task để kiểm tra nhanh")
    phase4.add_argument("--force", action="store_true", help="Bỏ qua cache feature và trích xuất lại")
    phase5 = subparsers.add_parser("phase5-train", help="Tuning, train/CV/validation/test cho Phase 4 feature tables")
    phase5.add_argument("--task", action="append", choices=["ripeness", "defect"],
                        help="Chọn task; mặc định chạy cả hai")
    predict = subparsers.add_parser("phase5-predict", help="Predict từ JSON mapping các feature Phase 4")
    predict.add_argument("--model", required=True, help="Model artifact .joblib")
    predict.add_argument("--features-json", required=True, help="JSON object gồm feature_name: value")
    subparsers.add_parser("list-datasets", help="Liệt kê nguồn đã cấu hình và số file đã import")
    remote = subparsers.add_parser("remote-files", help="Liệt kê file từ API Mendeley (cần OAuth token)")
    remote.add_argument("--dataset", required=True)
    return parser.parse_args()


def _selected(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    keys = args.dataset or list(config["datasets"])
    invalid = sorted(set(keys) - set(config["datasets"]))
    if invalid:
        raise ValueError(f"Dataset key không hợp lệ: {', '.join(invalid)}")
    return keys


def _download(config: dict[str, Any], keys: list[str]) -> int:
    downloader = MendeleyDownloader(config)
    failures = 0
    for index, key in enumerate(keys):
        try:
            files = downloader.download([key]).get(key, [])
            print(f"{key}: {len(files)} file đã có/tải mới")
        except AuthenticationRequired as exc:
            failures += 1
            LOGGER.error("%s: %s", key, exc)
            remaining = keys[index + 1:]
            if remaining:
                LOGGER.warning(
                    "Dừng tải API cho các nguồn còn lại sau lỗi xác thực; không gửi thêm request: %s",
                    ", ".join(remaining),
                )
            break
        except DownloadError as exc:
            failures += 1
            LOGGER.error("%s: %s", key, exc)
    return failures


def _download_public(config: dict[str, Any], keys: list[str]) -> int:
    downloader = MendeleyDownloader(config)
    failures = 0
    for index, key in enumerate(keys):
        try:
            target = downloader.download_public_zips([key]).get(key)
            print(f"{key}: {target if target else 'không có file tải'}")
        except AuthenticationRequired as exc:
            failures += 1
            LOGGER.error("%s: %s", key, exc)
            remaining = keys[index + 1:]
            if remaining:
                LOGGER.warning(
                    "Dừng tải public ZIP sau yêu cầu xác thực; không gửi thêm request: %s",
                    ", ".join(remaining),
                )
            break
        except DownloadError as exc:
            failures += 1
            LOGGER.error("%s: %s", key, exc)
    return failures


def _list_datasets(config: dict[str, Any]) -> None:
    for key, definition in config["datasets"].items():
        raw = project_path(config, f"data/raw/{key}")
        image_count = sum(1 for p in raw.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".ppm", ".pgm", ".pbm", ".pnm", ".ico"}) if raw.exists() else 0
        print(f"{key}: {image_count} tệp ảnh/mask trong raw; task={definition['task']}; {definition['url']}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    command = args.command
    if command == "list-datasets":
        _list_datasets(config)
        return 0
    if command == "remote-files":
        if args.dataset not in config["datasets"]:
            LOGGER.error("Dataset key không có trong config: %s", args.dataset)
            return 2
        try:
            files = MendeleyDownloader(config).list_remote(args.dataset)
        except DownloadError as exc:
            LOGGER.error("%s", exc)
            return 1
        for item in files:
            print(f"{item.get('_relative_folder', '')}/{item.get('filename', '')} [{item.get('id', '')}]")
        print(f"Tổng file: {len(files)}")
        return 0
    if command == "report":
        print(generate_report(config))
        return 0
    if command == "phase02":
        print(run_phase02(config))
        return 0
    if command == "phase1":
        print(run_phase1(
            config,
            dataset_keys=args.dataset,
            samples_per_group=args.samples_per_group,
            gray_world_enabled=False if args.no_gray_world else None,
            force=args.force,
        ))
        return 0
    if command == "phase2-segmentation":
        print(run_phase2(
            config,
            dataset_keys=args.dataset,
            per_group=args.samples_per_group,
            gray_world_enabled=False if args.no_gray_world else None,
            force=args.force,
            image_id=args.image_id,
            limit=args.limit,
        ))
        return 0
    if command == "annotate-fruit-masks":
        annotate_fruit_masks(config, args.limit)
        return 0
    if command == "import-fruit-masks":
        print(import_fruit_masks(config, args.input_dir))
        return 0
    if command == "approve-fruit-masks":
        manifest_path = approve_fruit_masks(config, args.image_id)
        from src.validation.phase07 import build_phase07_report, evaluate_approved_fruit_masks
        root = Path(config["_project_root"])
        evaluation = evaluate_approved_fruit_masks(root)
        report_path = build_phase07_report(root)
        print(json.dumps({"approved_manifest": str(manifest_path), "fruit_metrics": evaluation,
                          "phase07_report": str(report_path)}, ensure_ascii=False, indent=2, default=str))
        return 0
    if command == "phase3-kmeans":
        print(run_phase3(config, limit=args.limit, modes=args.mode, force=args.force))
        return 0
    if command == "phase4-features":
        print(run_phase4(config, tasks=args.task, limit=args.limit, force=args.force))
        return 0
    if command == "phase5-train":
        from src.models.phase5 import run_phase5
        print(run_phase5(config, tasks=args.task))
        return 0
    if command == "phase5-predict":
        import json
        from src.models.phase5 import predict_from_features
        model_path = project_path(config, args.model)
        values_path = project_path(config, args.features_json)
        values = json.loads(values_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("--features-json must contain a JSON object")
        print(json.dumps(predict_from_features(model_path, values), ensure_ascii=False))
        return 0
    keys = _selected(args, config)
    if command == "download":
        failures = _download(config, keys)
        return 1 if failures else 0
    if command == "download-public":
        failures = _download_public(config, keys)
        return 1 if failures else 0
    if command == "extract":
        print(extract_archives(config, keys))
    elif command == "inspect":
        print(inspect_datasets(config, keys))
    elif command == "validate":
        print(validate_manifest(config))
    elif command == "organize":
        print(organize_images(config))
    elif command == "split":
        print(prepare_splits(config, keys))
    elif command == "report":
        print(generate_report(config))
    elif command == "pipeline":
        failures = _download_public(config, keys)
        print(f"Download: {failures} nguồn chưa tải được; tiếp tục với file đã có/import thủ công.")
        print("Extract:", extract_archives(config, keys))
        print("Inspect:", inspect_datasets(config, keys))
        print("Validate:", validate_manifest(config))
        print("Organize:", organize_images(config))
        print("Split:", prepare_splits(config, keys))
        print("Report:", generate_report(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
