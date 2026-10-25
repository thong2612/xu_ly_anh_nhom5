from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.color.preprocess import (
    GrayWorldResult,
    decode_image,
    describe_channels,
    gray_world,
    rgb_to_hsv,
    rgb_to_lab,
    visible_rgb,
)
from src.utils.config import project_path


LOGGER = logging.getLogger("mangoresearch.phase1")
PIPELINE_VERSION = "phase1-color-v2"
STAT_METRICS = ("mean", "median", "std", "min", "max", "p1", "p5", "p25", "p50", "p75", "p95", "p99")
COLOR_CHANNELS = {"HSV": ("H", "S", "V"), "Lab": ("L*", "a*", "b*")}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(image_id: str, content_hash: str, gray_world_enabled: bool, preview_side: int) -> str:
    payload = f"{PIPELINE_VERSION}|{image_id}|{content_hash}|gw={int(gray_world_enabled)}|side={preview_side}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Phase 0.2 manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _eligible(row: dict[str, str]) -> bool:
    task = row.get("task", "")
    if task == "ripeness_classification":
        return _truthy(row.get("eligible_for_ripeness", "")) and not _truthy(row.get("ripeness_label_conflict", ""))
    if task == "defect_analysis":
        return _truthy(row.get("eligible_for_defect_analysis", ""))
    if task == "defect_segmentation":
        return _truthy(row.get("eligible_for_segmentation", ""))
    return False


def _label(row: dict[str, str]) -> str:
    if row.get("task") == "defect_segmentation":
        return row.get("mask_type", "") or row.get("defect_label", "") or ""
    return row.get("ripeness_label", "") or row.get("defect_label", "") or ""


def select_samples(
    rows: list[dict[str, str]], project_root: Path, dataset_keys: list[str], per_group: int, seed: int
) -> list[dict[str, Any]]:
    """Select deterministic, label-balanced samples, avoiding exact pixel duplicates."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") not in dataset_keys or not _eligible(row):
            continue
        path = project_path({"_project_root": str(project_root)}, row.get("image_path", ""))
        if path.is_file():
            groups[(row.get("dataset", ""), row.get("task", ""), _label(row))].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        candidates = list(groups[key])
        rng.shuffle(candidates)
        chosen: list[dict[str, str]] = []
        seen_pixels: set[str] = set()
        for row in candidates:
            pixel_hash = row.get("pixel_sha256", "") or row.get("image_id", "")
            if pixel_hash in seen_pixels:
                continue
            chosen.append(row)
            seen_pixels.add(pixel_hash)
            if len(chosen) >= per_group:
                break
        for row in chosen:
            record = dict(row)
            record["absolute_image_path"] = str(project_path({"_project_root": str(project_root)}, row["image_path"]))
            record["display_label"] = _label(row)
            selected.append(record)
    return selected


def _stats_for_rgb(rgb: np.ndarray, visible_mask: np.ndarray | None) -> tuple[dict[str, Any], dict[str, Any]]:
    hsv = rgb_to_hsv(rgb)
    hsv_stats = describe_channels(hsv, "HSV", visible_mask)
    del hsv
    lab = rgb_to_lab(rgb)
    lab_stats = describe_channels(lab, "Lab", visible_mask)
    del lab
    return hsv_stats, lab_stats


def _serialize_stats(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for channel, item in value.items():
        output[channel] = {
            key: (val.tolist() if isinstance(val, np.ndarray) else val)
            for key, val in item.items()
        }
    return output


def _restore_stats(value: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for channel, item in value.items():
        output[channel] = dict(item)
        output[channel]["histogram_counts"] = np.asarray(item["histogram_counts"], dtype=np.int64)
        output[channel]["histogram_edges"] = np.asarray(item["histogram_edges"], dtype=np.float32)
    return output


def _resize_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return rgb.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # INTER_AREA is suitable for reduction; RGB's dimensions and aspect ratio remain intact.
    import cv2

    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _process_one(sample: dict[str, Any], cache_dir: Path, gray_world_enabled: bool, preview_side: int, force: bool) -> tuple[str, list[dict[str, Any]], dict[str, np.ndarray], bool]:
    path = Path(sample["absolute_image_path"])
    source_hash = _hash_file(path)
    cache_key = _cache_key(sample["image_id"], source_hash, gray_world_enabled, preview_side)
    cache_path = cache_dir / f"{cache_key}.npz"
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as archive:
            payload = json.loads(str(archive["records_json"].item()))
            preview = {name: archive[name].copy() for name in archive.files if name.startswith("preview_")}
        return cache_key, payload, preview, True

    frame = decode_image(path)
    original = frame.rgb
    visible = None if frame.alpha is None else frame.alpha > 0
    if gray_world_enabled:
        balance = gray_world(original, frame.alpha)
    else:
        means = tuple(float(x) for x in original.reshape(-1, 3).mean(axis=0))
        balance = GrayWorldResult(original.copy(), (1.0, 1.0, 1.0), means, float(np.mean(means)), 0.0, 0.0, False, "disabled")
    balanced = balance.rgb if gray_world_enabled else original
    rows: list[dict[str, Any]] = []
    for experiment, color_image in (("E1", original), ("E2", balanced)):
        hsv_stats, lab_stats = _stats_for_rgb(color_image, visible)
        rows.append({"experiment": experiment, "HSV": _serialize_stats(hsv_stats), "Lab": _serialize_stats(lab_stats)})
        del hsv_stats, lab_stats

    # Full-resolution arrays are used for numeric summaries only. Cached preview
    # arrays are deliberately downsampled to keep storage bounded.
    display_original = _resize_rgb(visible_rgb(original, frame.alpha), preview_side)
    display_balanced = _resize_rgb(visible_rgb(balanced, frame.alpha), preview_side)
    preview: dict[str, np.ndarray] = {
        "preview_original": display_original,
        "preview_gray_world": display_balanced,
    }
    for suffix, display_rgb in (("e1", display_original), ("e2", display_balanced)):
        preview[f"preview_hsv_{suffix}"] = rgb_to_hsv(display_rgb)
        preview[f"preview_lab_{suffix}"] = rgb_to_lab(display_rgb)
    payload = {
        "gray_world": {
            "enabled": gray_world_enabled,
            "gains": list(balance.gains),
            "channel_means": list(balance.channel_means),
            "target_mean": balance.target_mean,
            "clipping_fraction": balance.clipping_fraction,
            "clipped_channel_value_fraction": balance.clipped_channel_value_fraction,
            "applied": balance.applied,
            "reason": balance.reason,
        },
        "experiments": rows,
        "native_height": int(original.shape[0]),
        "native_width": int(original.shape[1]),
        "preview_height": int(display_original.shape[0]),
        "preview_width": int(display_original.shape[1]),
        "visible_pixel_count": int(np.count_nonzero(visible)) if visible is not None else int(original.shape[0] * original.shape[1]),
        "rgba": frame.alpha is not None,
        "source_sha256": source_hash,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, records_json=np.asarray(json.dumps(payload)), **preview)
    temporary.replace(cache_path)
    return cache_key, payload, preview, False


def _flat_stats_record(sample: dict[str, Any], cache_key: str, payload: dict[str, Any], experiment: dict[str, Any], cache_hit: bool) -> dict[str, Any]:
    gw = payload["gray_world"]
    row: dict[str, Any] = {
        "cache_key": cache_key,
        "image_id": sample["image_id"],
        "dataset": sample["dataset"],
        "task": sample["task"],
        "label": sample["display_label"],
        "image_path": sample["image_path"],
        "split": sample.get("phase02_split", "") or sample.get("split", ""),
        "experiment": experiment["experiment"],
        "gray_world_enabled": str(gw["enabled"]).lower(),
        "gray_world_applied": gw["applied"],
        "gray_world_reason": gw["reason"],
        "gain_r": gw["gains"][0],
        "gain_g": gw["gains"][1],
        "gain_b": gw["gains"][2],
        "input_mean_r": gw["channel_means"][0],
        "input_mean_g": gw["channel_means"][1],
        "input_mean_b": gw["channel_means"][2],
        "gray_world_clipping_fraction": gw["clipping_fraction"],
        "gray_world_clipped_channel_value_fraction": gw["clipped_channel_value_fraction"],
        "width": payload["native_width"],
        "height": payload["native_height"],
        "visible_pixel_count": payload["visible_pixel_count"],
        "rgba": payload["rgba"],
        "source_sha256": payload["source_sha256"],
        "cache_hit": cache_hit,
    }
    for space in ("HSV", "Lab"):
        stats = _restore_stats(experiment[space])
        for channel, values in stats.items():
            prefix = f"{space}_{channel}"
            for metric in STAT_METRICS:
                row[f"{prefix}_{metric}"] = values[metric]
            if space == "HSV" and channel == "H":
                edges = np.asarray(values["histogram_edges"], dtype=np.float64)
                counts = np.asarray(values["histogram_counts"], dtype=np.float64)
                centers = np.deg2rad((edges[:-1] + edges[1:]) / 2.0)
                sine = float(np.sum(np.sin(centers) * counts))
                cosine = float(np.sum(np.cos(centers) * counts))
                row["HSV_H_circular_mean_approx"] = float(np.rad2deg(np.arctan2(sine, cosine)) % 360.0)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _histogram_records(samples: list[dict[str, Any]], payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for sample in samples:
        payload = payloads[sample["image_id"]]
        for experiment in payload["experiments"]:
            for space in ("HSV", "Lab"):
                stats = experiment[space]
                for channel, item in stats.items():
                    result.append({
                        "image_id": sample["image_id"],
                        "dataset": sample["dataset"],
                        "task": sample["task"],
                        "label": sample["display_label"],
                        "experiment": experiment["experiment"],
                        "color_space": space,
                        "channel": channel,
                        "histogram_edges": json.dumps(item["histogram_edges"]),
                        "histogram_counts": json.dumps(item["histogram_counts"]),
                    })
    return result


def _plot_sample_preview(sample: dict[str, Any], payload: dict[str, Any], arrays: dict[str, np.ndarray], target: Path) -> None:
    original = arrays["preview_original"]
    balanced = arrays["preview_gray_world"]
    fig, axes = plt.subplots(5, 3, figsize=(16, 20), constrained_layout=True)
    axes[0, 0].imshow(original)
    axes[0, 0].set_title("RGB original")
    axes[0, 1].imshow(balanced)
    axes[0, 1].set_title("Gray World" if payload["gray_world"]["enabled"] else "Gray World disabled")
    axes[0, 0].axis("off")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    gw = payload["gray_world"]
    axes[0, 2].text(0.02, 0.98, "Sample details\n"
                    f"dataset: {sample['dataset']}\ntask: {sample['task']}\nlabel: {sample['display_label']}\n"
                    f"size: {payload['native_width']}×{payload['native_height']}\n"
                    f"visible pixels: {payload['visible_pixel_count']}\n"
                    f"gains RGB: {np.round(gw['gains'], 4).tolist()}\n"
                    f"clipped pixels: {gw['clipping_fraction']:.6%}\n"
                    f"clipped channel values: {gw['clipped_channel_value_fraction']:.6%}\n"
                    f"Gray World: {gw['reason']}\n"
                    "Statistics include the full frame and background; no fruit mask was applied.",
                    va="top", fontsize=10, wrap=True)
    channels = [
        ("HSV H", arrays["preview_hsv_e1"][:, :, 0], arrays["preview_hsv_e2"][:, :, 0], 0, 360),
        ("HSV S", arrays["preview_hsv_e1"][:, :, 1], arrays["preview_hsv_e2"][:, :, 1], 0, 1),
        ("HSV V", arrays["preview_hsv_e1"][:, :, 2], arrays["preview_hsv_e2"][:, :, 2], 0, 1),
        ("CIELAB L*", arrays["preview_lab_e1"][:, :, 0], arrays["preview_lab_e2"][:, :, 0], 0, 100),
        ("CIELAB a*", arrays["preview_lab_e1"][:, :, 1], arrays["preview_lab_e2"][:, :, 1], -128, 128),
        ("CIELAB b*", arrays["preview_lab_e1"][:, :, 2], arrays["preview_lab_e2"][:, :, 2], -128, 128),
    ]
    for index, (title, before, after, low, high) in enumerate(channels):
        row_index, column_index = 1 + index // 3, index % 3
        channel_pair = np.hstack((before, after))
        axes[row_index, column_index].imshow(channel_pair, cmap="gray", vmin=low, vmax=high)
        axes[row_index, column_index].axvline(before.shape[1] - 0.5, color="red", linewidth=1)
        axes[row_index, column_index].set_title(f"{title} — E1 trái / E2 phải, kênh đơn")
    for row in (1, 2):
        for axis in axes[row]:
            axis.axis("off")

    histogram_panels = (("HSV", "H"), ("HSV", "S"), ("HSV", "V"),
                        ("Lab", "L*"), ("Lab", "a*"), ("Lab", "b*"))
    for index, (space, channel) in enumerate(histogram_panels):
        row_index, column_index = 3 + index // 3, index % 3
        axis = axes[row_index, column_index]
        for experiment in payload["experiments"]:
            exp_name = experiment["experiment"]
            item = experiment[space][channel]
            edges = np.asarray(item["histogram_edges"], dtype=np.float32)
            counts = np.asarray(item["histogram_counts"], dtype=np.float64)
            normalized = counts / max(1.0, counts.sum())
            centers = (edges[:-1] + edges[1:]) / 2
            axis.plot(centers, normalized, label=exp_name, linewidth=1.2,
                      linestyle="-" if exp_name == "E1" else "--")
        axis.set_title(f"{space} {channel} — E1 liền / E2 đứt")
        axis.set_xlabel("Giá trị kênh theo thang đã ghi")
        axis.set_ylabel("Tỷ lệ pixel")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(f"{sample['dataset']} | {sample['display_label']} | {sample['image_id']}\n"
                 "E1 = ảnh gốc; E2 = Gray World | HSV: H°/S,V unit; Lab: L*[0,100], a*/b* signed", fontsize=13)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)


def _aggregate_histograms(samples: list[dict[str, Any]], payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[np.ndarray]] = defaultdict(list)
    edge_lookup: dict[tuple[str, str], np.ndarray] = {}
    for sample in samples:
        if sample["dataset"] != "mango_ripeness":
            continue
        payload = payloads[sample["image_id"]]
        for exp in payload["experiments"]:
            for space in ("HSV", "Lab"):
                for channel, item in exp[space].items():
                    key = (sample["display_label"], exp["experiment"], space, channel, sample["dataset"])
                    counts = np.asarray(item["histogram_counts"], dtype=np.float64)
                    grouped[key].append(counts / max(1.0, counts.sum()))
                    edge_lookup[(space, channel)] = np.asarray(item["histogram_edges"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for (label, experiment, space, channel, dataset), histograms in sorted(grouped.items()):
        mean_histogram = np.mean(np.stack(histograms), axis=0)
        rows.append({"dataset": dataset, "label": label, "experiment": experiment,
                     "color_space": space, "channel": channel,
                     "image_count": len(histograms),
                     "histogram_edges": json.dumps(edge_lookup[(space, channel)].tolist()),
                     "mean_per_image_fraction": json.dumps(mean_histogram.tolist())})
    return rows


def _plot_ripeness_histograms(aggregate: list[dict[str, Any]], target: Path) -> None:
    labels = ("unripe", "partially_ripe", "ripe")
    colors = {"unripe": "#1b9e77", "partially_ripe": "#d95f02", "ripe": "#7570b3"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for axis, space, channel in zip(axes.ravel(), ("HSV", "HSV", "HSV", "Lab", "Lab", "Lab"), COLOR_CHANNELS["HSV"] + COLOR_CHANNELS["Lab"]):
        for label in labels:
            for exp, linestyle in (("E1", "-"), ("E2", "--")):
                row = next((item for item in aggregate if item["label"] == label and item["experiment"] == exp and item["color_space"] == space and item["channel"] == channel), None)
                if not row:
                    continue
                edges = np.asarray(json.loads(row["histogram_edges"]), dtype=np.float32)
                fractions = np.asarray(json.loads(row["mean_per_image_fraction"]), dtype=np.float64)
                centers = (edges[:-1] + edges[1:]) / 2
                axis.plot(centers, fractions, color=colors[label], linestyle=linestyle,
                          label=f"{label} {exp}", linewidth=1.2)
        axis.set_title(f"{space} {channel}")
        axis.set_ylabel("Mean per-image pixel fraction")
        axis.set_xlabel("Channel value")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle("Dataset A sampled ripeness groups — E1 solid, E2 dashed", fontsize=14)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def _plot_case_examples(samples: list[dict[str, Any]], payloads: dict[str, dict[str, Any]], previews: dict[str, dict[str, np.ndarray]], preview_paths: dict[str, Path], target: Path, gain_bounds: tuple[float, float], clip_limit: float) -> list[str]:
    candidates: list[tuple[float, bool, dict[str, Any]]] = []
    gain_low, gain_high = gain_bounds
    for sample in samples:
        gw = payloads[sample["image_id"]]["gray_world"]
        gain_deviation = max(abs(float(gain) - 1.0) for gain in gw["gains"])
        score = gain_deviation + float(gw["clipping_fraction"]) * 5.0
        flagged = (any(float(gain) < gain_low or float(gain) > gain_high for gain in gw["gains"])
                   or float(gw["clipping_fraction"]) > clip_limit)
        candidates.append((score, flagged, sample))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0])
    good = ("Low-change", candidates[0][2])
    ordinary = [candidate for candidate in candidates if not candidate[1]] or candidates
    average = ("Middle-change", ordinary[len(ordinary) // 2][2])
    flagged = [candidate for candidate in candidates if candidate[1]]
    abnormal = ("Abnormal (threshold flagged)", flagged[-1][2]) if flagged else ("Highest-change (none flagged)", candidates[-1][2])
    selected_cases: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for label, sample in (good, average, abnormal):
        if sample["image_id"] not in seen:
            selected_cases.append((label, sample))
            seen.add(sample["image_id"])
    fig, axes = plt.subplots(len(selected_cases), 3, figsize=(12, 4 * len(selected_cases)), squeeze=False, constrained_layout=True)
    for row_index, (case_label, sample) in enumerate(selected_cases):
        image_id = sample["image_id"]
        image = previews[image_id]["preview_original"]
        balanced = previews[image_id]["preview_gray_world"]
        gw = payloads[image_id]["gray_world"]
        axes[row_index, 0].imshow(image)
        short_label = "Low change" if case_label == "Low-change" else "Typical" if case_label == "Middle-change" else "Abnormal" if case_label == "Abnormal (threshold flagged)" else "Highest change (no flag)"
        axes[row_index, 0].set_title(f"{short_label} | {sample['dataset']} / {sample['display_label']}", fontsize=11)
        axes[row_index, 1].imshow(balanced)
        axes[row_index, 1].set_title("Gray World")
        axes[row_index, 2].axis("off")
        axes[row_index, 2].text(0, 1, f"id: {image_id}\ngains: {np.round(gw['gains'], 4).tolist()}\n"
                                f"clip fraction: {gw['clipping_fraction']:.6%}\nreason: {gw['reason']}\n"
                                f"score: {max(abs(float(g) - 1.0) for g in gw['gains']) + float(gw['clipping_fraction']) * 5.0:.4f}",
                                va="top")
        axes[row_index, 0].axis("off")
        axes[row_index, 1].axis("off")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return [str(preview_paths[sample["image_id"]]) for _, sample in selected_cases]


def _write_report(report_path: Path, manifest_rows: list[dict[str, str]], samples: list[dict[str, Any]],
                  stats_rows: list[dict[str, Any]], failure_rows: list[dict[str, Any]],
                  aggregate: list[dict[str, Any]], preview_paths: dict[str, Path], cases_path: Path,
                  histogram_path: Path, enabled: bool, cache_hits: int, report_root: Path,
                  gain_bounds: tuple[float, float], clip_threshold: float) -> None:
    eligible_counts = Counter((row["dataset"], row["task"], _label(row)) for row in manifest_rows if _eligible(row))
    sampled_counts = Counter((row["dataset"], row["task"], row["display_label"]) for row in samples)
    a_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in stats_rows:
        if row["dataset"] == "mango_ripeness":
            for channel in ("HSV_H_circular_mean_approx", "HSV_S_mean", "HSV_V_mean", "Lab_L*_mean", "Lab_a*_mean", "Lab_b*_mean"):
                a_stats[f"{row['label']} / {row['experiment']}"][channel].append(float(row[channel]))
    by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in stats_rows:
        by_image[row["image_id"]][row["experiment"]] = row
    change_groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"deltas": defaultdict(list), "gains": [[], [], []], "clips": [], "channel_clips": []})
    channels = ("HSV_H", "HSV_S", "HSV_V", "Lab_L*", "Lab_a*", "Lab_b*")
    for sample in samples:
        paired = by_image.get(sample["image_id"], {})
        if "E1" not in paired or "E2" not in paired:
            continue
        group = change_groups[(sample["dataset"], sample["task"], sample["display_label"])]
        for channel in channels:
            if channel == "HSV_H":
                difference = float(paired["E2"]["HSV_H_circular_mean_approx"]) - float(paired["E1"]["HSV_H_circular_mean_approx"])
                difference = (difference + 180.0) % 360.0 - 180.0
            else:
                difference = float(paired["E2"][f"{channel}_mean"]) - float(paired["E1"][f"{channel}_mean"])
            group["deltas"][channel].append(difference)
        for index, key in enumerate(("gain_r", "gain_g", "gain_b")):
            group["gains"][index].append(float(paired["E1"][key]))
        group["clips"].append(float(paired["E1"]["gray_world_clipping_fraction"]))
        group["channel_clips"].append(float(paired["E1"]["gray_world_clipped_channel_value_fraction"]))
    e1_rows = [rows["E1"] for rows in by_image.values() if "E1" in rows]
    flagged_rows = [row for row in e1_rows if
                    any(float(row[key]) < gain_bounds[0] or float(row[key]) > gain_bounds[1] for key in ("gain_r", "gain_g", "gain_b"))
                    or float(row["gray_world_clipping_fraction"]) > clip_threshold]
    gain_summary = []
    for key, channel in zip(("gain_r", "gain_g", "gain_b"), ("R", "G", "B")):
        values = [float(row[key]) for row in e1_rows]
        if values:
            gain_summary.append(f"{channel}: mean {np.mean(values):.4f}, std {np.std(values):.4f}, range [{min(values):.4f}, {max(values):.4f}]")
    clipping_values = [float(row["gray_world_clipping_fraction"]) for row in e1_rows]
    lines = [
        "# MangoResearch — Phase 1 color preprocessing report", "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
        "## Scope and reproducibility", "",
        f"This run used {len(samples)} deterministic samples selected from the Phase 0.2 clean manifest; {len(stats_rows)} image/experiment rows were summarized and {len(failure_rows)} selected images failed processing.",
        f"Gray World configured: **{enabled}**. Cache hits: {cache_hits}/{len(samples)}. Sample seed and per-group count are recorded in `data/metadata/phase1_sample_manifest.csv`.",
        "E1 is original RGB → HSV/CIELAB. E2 is Gray World RGB → HSV/CIELAB. E2 equals the source when Gray World is disabled or skipped for a near-zero channel.",
        "All color statistics cover the full frame, including background; no fruit mask was used. They are not mango-only color features.",
        "HSV is stored as float32 H [0,360), S/V [0,1]. OpenCV float CIELAB is stored as float32 L* [0,100] and signed a*/b*. Preview channel panels show E1 on the left and E2 on the right as single-channel grayscale mappings, not RGB renderings.", "",
        "## Dataset coverage", "", "| Dataset / task / label | Eligible in Phase 0.2 manifest | Sampled this run |", "|---|---:|---:|"
    ]
    for key in sorted(set(eligible_counts) | set(sampled_counts)):
        lines.append(f"| {' / '.join(key)} | {eligible_counts.get(key, 0)} | {sampled_counts.get(key, 0)} |")
    lines.extend(["", "Dataset A clean manifest totals are 748 images: 249 unripe, 249 partially ripe, 250 ripe. This Phase 1 run uses a balanced subset, not all 748 images.",
                  "", "## Dataset A color means by ripeness sample", "",
                  "Values below are the unweighted mean of per-image channel means in the sampled group. H uses a circular mean approximated from the 72-bin histogram to avoid hue wraparound. Full channel statistics and percentiles are in `reports/phase1_image_stats.csv`.",
                  "", "| Label / experiment | H circular mean (°) | S mean | V mean | L* mean | a* mean | b* mean |", "|---|---:|---:|---:|---:|---:|---:|"])
    for key in sorted(a_stats):
        vals = a_stats[key]
        entries = [float(np.mean(vals[channel])) if vals.get(channel) else float("nan")
                   for channel in ("HSV_H_circular_mean_approx", "HSV_S_mean", "HSV_V_mean", "Lab_L*_mean", "Lab_a*_mean", "Lab_b*_mean")]
        lines.append(f"| {key} | " + " | ".join(f"{value:.4f}" for value in entries) + " |")
    lines.extend(["", "Dataset A label histograms: [Dataset A ripeness histograms](" + histogram_path.relative_to(report_root.parent.parent).as_posix() + ").", "",
                  "## Gray World stability", "",
                  f"The run examined {len(samples)} images. Gain/clipping metrics are repeated in both experiment rows in `reports/phase1_image_stats.csv`; E1/E2 channel differences can be derived directly from the same table.",
                  f"Gain bounds were [{gain_bounds[0]}, {gain_bounds[1]}]; clipping review threshold was {clip_threshold:.2%} of pixels with any channel clipped. {len(flagged_rows)}/{len(e1_rows)} images were flagged. Pixel clipping and channel-value clipping are separately recorded in the statistics CSV.",
                  "Gain summary: " + ("; ".join(gain_summary) if gain_summary else "no successful samples") + ".",
                  (f"Clipped-pixel fraction across samples: mean {np.mean(clipping_values):.4%}, median {np.median(clipping_values):.4%}, max {max(clipping_values):.4%}." if clipping_values else "Clipped-pixel fraction: no successful samples."),
                  "", "### Mean E2−E1 color change by group", "",
                  "Each delta is the unweighted mean across sampled images; ΔH is the shortest circular shift of approximate hue means and other channels use arithmetic mean differences.",
                  "", "| Dataset / task / label | ΔH circular (°) | ΔS | ΔV | ΔL* | Δa* | Δb* | mean gain RGB | median clipped-pixel fraction | max clipped-pixel fraction |", "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"])
    for (dataset, task, label), group in sorted(change_groups.items()):
        deltas = [float(np.mean(group["deltas"][channel])) for channel in channels]
        gains = [float(np.mean(values)) if values else 1.0 for values in group["gains"]]
        clip_median = float(np.median(group["clips"])) if group["clips"] else 0.0
        clip_max = max(group["clips"]) if group["clips"] else 0.0
        lines.append(f"| {dataset} / {task} / {label} | " + " | ".join(f"{delta:.4f}" for delta in deltas)
                     + f" | {np.round(gains, 4).tolist()} | {clip_median:.4%} | {clip_max:.4%} |")
    lines.extend(["", f"[Low-change, middle-change, and abnormal (or highest-change if none crossed threshold) examples](<{cases_path.relative_to(report_root.parent.parent).as_posix()}>).", "",
                  "## Previews", ""])
    for sample in samples:
        preview_rel = preview_paths[sample["image_id"]].relative_to(report_root.parent.parent).as_posix()
        lines.append(f"- {sample['dataset']} / {sample['display_label']} / `{sample['image_id']}`: [preview](<{preview_rel}>).")
    lines.extend(["", "## Data quality and limitations", "",
                  f"Images failing during this run: {len(failure_rows)}. Failures are listed in `reports/phase1_failures.csv`; source rows and raw files were not changed.",
                  "Gray World assumes the scene-wide channel means should agree. Mango images contain colored backgrounds, lighting gradients, and fruit color variation, so the adjustment may shift real color cues. This phase reports color changes only and makes no accuracy claim.",
                  "Only eligible rows in the Phase 0.2 manifest were sampled. Dataset A ripeness conflicts remain excluded there. Existing Phase 0.2 split assignments are read for traceability and were not modified. The split's leakage caveats from Phase 0.2 remain in force.",
                  "", "## Phase 2 readiness", "",
                  "The pipeline produces reproducible float32 HSV/CIELAB statistics, controlled Gray World comparisons, and lossless compressed NumPy preview caches linked to original IDs. It is ready to feed image-level color data into Phase 2 after the research team decides whether and how to mask the fruit; current statistics include backgrounds.",
                  "No segmentation, K-Means, feature selection, or model training was performed.", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_phase1(config: dict[str, Any], dataset_keys: list[str] | None = None,
               samples_per_group: int | None = None, gray_world_enabled: bool | None = None,
               force: bool = False) -> Path:
    """Run Phase 1 color analysis on a deterministic subset of Phase 0.2 clean data."""
    phase1 = config.get("phase1", {})
    project_root = Path(config["_project_root"])
    manifest_path = project_path(config, phase1.get("manifest_path", "data/metadata/phase02_manifest.csv"))
    cache_dir = project_path(config, phase1.get("cache_dir", "data/processed/phase1_cache"))
    report_root = project_path(config, phase1.get("output_dir", "reports/phase1"))
    report_path = project_path(config, phase1.get("report_path", "reports/phase01_report.md"))
    enabled = bool(phase1.get("gray_world_enabled", True)) if gray_world_enabled is None else gray_world_enabled
    per_group = int(samples_per_group if samples_per_group is not None else phase1.get("samples_per_group", 2))
    seed = int(phase1.get("seed", 20261023))
    preview_side = int(phase1.get("preview_max_side", 512))
    gain_bounds = tuple(float(value) for value in phase1.get("abnormal_gain_bounds", [0.7, 1.3]))
    clip_threshold = float(phase1.get("abnormal_clipping_fraction", 0.01))
    if per_group < 1 or preview_side < 64:
        raise ValueError("samples_per_group must be >= 1 and preview_max_side >= 64")
    manifest_rows = _read_manifest(manifest_path)
    keys = dataset_keys or list(config["datasets"])
    invalid = set(keys) - set(config["datasets"])
    if invalid:
        raise ValueError(f"Unknown dataset key(s): {', '.join(sorted(invalid))}")
    samples = select_samples(manifest_rows, project_root, keys, per_group, seed)
    if not samples:
        raise ValueError("No eligible Phase 0.2 images found for the requested dataset selection")
    LOGGER.info("Phase 1: selected %d eligible image(s) from %d manifest rows", len(samples), len(manifest_rows))
    stats_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    previews: dict[str, dict[str, np.ndarray]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    cache_hits = 0
    preview_dir = report_root / "previews"
    preview_paths: dict[str, Path] = {}
    sample_manifest_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        try:
            LOGGER.info("Processing %d/%d: %s (%s)", index, len(samples), sample["image_id"], sample["dataset"])
            key, payload, arrays, cache_hit = _process_one(sample, cache_dir, enabled, preview_side, force)
            cache_hits += int(cache_hit)
            payloads[sample["image_id"]] = payload
            previews[sample["image_id"]] = arrays
            for experiment in payload["experiments"]:
                stats_rows.append(_flat_stats_record(sample, key, payload, experiment, cache_hit))
            preview_path = preview_dir / f"{sample['dataset']}__{sample['image_id']}.png"
            _plot_sample_preview(sample, payload, arrays, preview_path)
            preview_paths[sample["image_id"]] = preview_path
            sample_manifest_rows.append({
                "image_id": sample["image_id"], "dataset": sample["dataset"], "task": sample["task"],
                "label": sample["display_label"], "source_image_path": sample["image_path"],
                "source_sha256": payload["source_sha256"], "phase02_split": sample.get("phase02_split", ""),
                "cache_key": key, "cache_path": str((cache_dir / f"{key}.npz").relative_to(project_root)),
                "preview_path": str(preview_path.relative_to(project_root)), "native_width": payload["native_width"],
                "native_height": payload["native_height"], "gray_world_enabled": str(enabled).lower(),
            })
        except Exception as exc:  # retain an auditable row and continue with remaining samples
            LOGGER.exception("Failed to process %s", sample.get("image_id", "unknown"))
            failures.append({"image_id": sample.get("image_id", ""), "dataset": sample.get("dataset", ""),
                             "image_path": sample.get("image_path", ""), "error_type": type(exc).__name__, "error": str(exc)})

    output_dir = project_path(config, "reports")
    _write_csv(output_dir / "phase1_image_stats.csv", stats_rows)
    _write_csv(output_dir / "phase1_failures.csv", failures)
    _write_csv(project_path(config, "data/metadata/phase1_sample_manifest.csv"), sample_manifest_rows)
    histogram_rows = _histogram_records(samples=[s for s in samples if s["image_id"] in payloads], payloads=payloads)
    _write_csv(output_dir / "phase1_histograms.csv", histogram_rows)
    aggregate = _aggregate_histograms([s for s in samples if s["image_id"] in payloads], payloads)
    _write_csv(output_dir / "phase1_aggregate_histograms.csv", aggregate)
    histogram_path = report_root / "dataset_a_ripeness_histograms.png"
    if aggregate:
        _plot_ripeness_histograms(aggregate, histogram_path)
    else:
        histogram_path.write_bytes(b"")
    cases_path = report_root / "gray_world_case_examples.png"
    _plot_case_examples([s for s in samples if s["image_id"] in payloads], payloads, previews,
                        preview_paths, cases_path, gain_bounds, clip_threshold)
    _write_report(report_path, manifest_rows, [s for s in samples if s["image_id"] in payloads], stats_rows,
                  failures, aggregate, preview_paths, cases_path, histogram_path, enabled, cache_hits,
                  report_root, gain_bounds, clip_threshold)
    LOGGER.info("Phase 1 outputs written: %s", report_path)
    return report_path
