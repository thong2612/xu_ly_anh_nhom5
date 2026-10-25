from __future__ import annotations

import argparse
import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from src.color.preprocess import decode_image
from src.validation.phase07 import (
    KMEANS_REVIEW_PATH, _mask_image, evaluate_reviewed_kmeans_pairs, set_kmeans_pair_review,
)
from src.validation.kmeans_prereview import run_kmeans_prereview

WARNING_LABELS = {
    "source_image_unreadable": "không đọc được ảnh nguồn",
    "mask_not_png": "mask nguồn không phải PNG",
    "mask_unreadable_or_missing": "không đọc được mask nguồn",
    "mask_not_single_channel": "mask không phải ảnh đơn kênh",
    "mask_not_uint8": "mask không ở uint8",
    "mask_size_mismatch": "kích thước mask khác ảnh",
    "mask_not_binary_0_255": "mask có giá trị ngoài 0/255",
    "mask_empty": "mask rỗng",
    "phase2_fruit_mask_path_conflict": "metadata có nhiều đường dẫn fruit mask khác nhau",
    "phase2_fruit_mask_invalid_or_unavailable": "fruit mask heuristic không hợp lệ hoặc thiếu",
    "phase2_fruit_mask_unavailable": "không có fruit mask heuristic",
    "mask_partially_outside_heuristic_fruit_mask": "một phần đáng kể mask nằm ngoài fruit mask heuristic",
    "mask_extends_outside_heuristic_fruit_mask": "một phần mask nằm ngoài fruit mask heuristic",
    "mask_area_extremely_small": "diện tích mask rất nhỏ so với ảnh",
    "mask_area_extremely_large": "diện tích mask rất lớn so với ảnh",
    "mask_touches_image_border": "mask chạm rìa ảnh",
}


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy danh sách 50 cặp: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Danh sách review chưa có cặp nào. Hãy chạy lệnh phase07 prepare trước.")
    return rows


def _blend(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    selected = mask > 0
    if selected.any():
        output[selected] = output[selected] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


class KMeansReviewApp:
    def __init__(self, root: Path) -> None:
        self.root_path = root.resolve()
        self.manifest_path = self.root_path / KMEANS_REVIEW_PATH
        self.session_path = self.root_path / "data/metadata/phase07_kmeans_review_session.json"
        self.prereview_result = run_kmeans_prereview(self.root_path)
        self.rows = _read_review_rows(self.manifest_path)
        self.index = self._restore_index()
        self.root = tk.Tk()
        self.root.title("MangoResearch — Review 50 cặp ảnh/mask khuyết tật")
        self.root.geometry("1320x900")
        self.root.minsize(980, 680)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.photo_refs: list[ImageTk.PhotoImage] = []
        self.zoom = tk.DoubleVar(value=1.15)
        self.overlay_alpha = tk.DoubleVar(value=0.48)
        self.reviewer = tk.StringVar()
        self.progress_text = tk.StringVar()
        self.status_text = tk.StringVar()
        self._build()
        self._show_current()

    def _restore_index(self) -> int:
        if self.session_path.is_file():
            try:
                image_id = json.loads(self.session_path.read_text(encoding="utf-8")).get("current_image_id")
                for i, row in enumerate(self.rows):
                    if row.get("image_id") == image_id:
                        return i
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        return 0

    def _save_session(self) -> None:
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"current_image_id": self.rows[self.index].get("image_id", ""), "current_index": self.index}
        temp = self.session_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.session_path)

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.progress_text, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.status_text, foreground="#344054").pack(side="left", padx=18)
        ttk.Button(top, text="← Trước", command=self.previous).pack(side="right", padx=4)
        ttk.Button(top, text="Tiếp →", command=self.next).pack(side="right", padx=4)
        ttk.Button(top, text="Chạy đánh giá approved", command=self.evaluate).pack(side="right", padx=12)

        tools = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        tools.pack(fill="x")
        ttk.Label(tools, text="Phóng to").pack(side="left")
        ttk.Scale(tools, from_=0.65, to=2.0, variable=self.zoom, command=lambda _v: self._show_current()).pack(side="left", fill="x", expand=True, padx=(6, 18))
        ttk.Label(tools, text="Độ đậm overlay").pack(side="left")
        ttk.Scale(tools, from_=0.15, to=0.8, variable=self.overlay_alpha, command=lambda _v: self._show_current()).pack(side="left", fill="x", expand=True, padx=6)

        area = ttk.Frame(self.root)
        area.pack(fill="both", expand=True, padx=10)
        self.canvas = tk.Canvas(area, highlightthickness=0)
        scroll = ttk.Scrollbar(area, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(area, orient="horizontal", command=self.canvas.xview)
        self.panel_frame = ttk.Frame(self.canvas)
        self.panel_frame.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.panel_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal.set)
        horizontal.pack(side="bottom", fill="x")
        scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)

        review = ttk.LabelFrame(self.root, text="Quyết định của người review (được lưu ngay sau khi bấm)", padding=8)
        review.pack(fill="x", padx=10, pady=(6, 10))
        row = ttk.Frame(review)
        row.pack(fill="x")
        ttk.Label(row, text="Người review:").pack(side="left")
        self.reviewer_entry = ttk.Entry(row, textvariable=self.reviewer, width=26)
        self.reviewer_entry.pack(side="left", padx=(4, 14))
        ttk.Label(row, text="Ghi chú quan sát (ví dụ: mask lệch, bỏ sót vùng, bóng đổ, màu vỏ tự nhiên, fruit mask):").pack(side="left")
        self.notes = tk.Text(review, height=2, wrap="word")
        self.notes.pack(fill="x", pady=(5, 7))
        actions = ttk.Frame(review)
        actions.pack(fill="x")
        ttk.Button(actions, text="Approved — đã xác nhận mask", command=lambda: self.decide("approved")).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Rejected — mask sai/lệch", command=lambda: self.decide("rejected")).pack(side="left", padx=6)
        ttk.Button(actions, text="Uncertain — chưa đủ cơ sở", command=lambda: self.decide("uncertain")).pack(side="left", padx=6)
        ttk.Label(actions, text="←/→ chuyển ảnh; trạng thái chỉ đổi khi bạn bấm một trong ba nút.", foreground="#475467").pack(side="right")
        self.root.bind_all("<Left>", self._left_key)
        self.root.bind_all("<Right>", self._right_key)

    def _editing_text(self) -> bool:
        focus = self.root.focus_get()
        return focus is self.notes or focus is getattr(self, "reviewer_entry", None)

    def _left_key(self, _event: tk.Event) -> None:
        if not self._editing_text():
            self.previous()

    def _right_key(self, _event: tk.Event) -> None:
        if not self._editing_text():
            self.next()

    def _mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _current(self) -> dict[str, str]:
        return self.rows[self.index]

    def _read_mask_for_view(self, field: str, shape: tuple[int, int]) -> np.ndarray | None:
        value = self._current().get(field, "")
        if not value:
            return None
        return _mask_image(self.root_path / Path(value), shape)

    def _show_current(self) -> None:
        row = self._current()
        self._save_session()
        statuses = {name: sum((r.get("review_status") or "pending") == name for r in self.rows)
                    for name in ("approved", "rejected", "uncertain", "pending")}
        self.progress_text.set(f"Cặp {self.index + 1}/{len(self.rows)}  •  {row.get('image_id', '')}  •  "
                               f"Approved {statuses['approved']} | Rejected {statuses['rejected']} | "
                               f"Uncertain {statuses['uncertain']} | Pending {statuses['pending']}")
        self.status_text.set(f"Trạng thái hiện tại: {row.get('review_status') or 'pending'} | "
                             f"Chọn mẫu: {row.get('selection_reason', '')} | "
                             f"đủ điều kiện Accuracy chính thức: {row.get('official_accuracy_eligible', 'false')}")
        self.reviewer.set(row.get("reviewer", ""))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", row.get("review_notes", ""))
        for child in self.panel_frame.winfo_children():
            child.destroy()
        self.photo_refs.clear()
        try:
            rgb = decode_image(self.root_path / Path(row["source_image_path"])).rgb
            image_error = ""
        except Exception as exc:
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            image_error = f"Không đọc được ảnh: {exc}"
        shape = rgb.shape[:2]
        alpha = float(self.overlay_alpha.get())
        source_mask = self._read_mask_for_view("ground_truth_mask_path", shape)
        lab_mask = self._read_mask_for_view("lab_k3_prediction_path", shape)
        hsv_mask = self._read_mask_for_view("hsv_k5_prediction_path", shape)
        max_edge = max(180, int(360 * float(self.zoom.get())))

        def add_view(parent: ttk.Frame, index: int, columns: int, title: str,
                     view: np.ndarray | None, note: str = "") -> None:
            panel = ttk.LabelFrame(parent, text=title, padding=5)
            panel.grid(row=index // columns, column=index % columns, sticky="nsew", padx=5, pady=5)
            parent.columnconfigure(index % columns, weight=1)
            if view is None:
                placeholder = Image.new("RGB", (max_edge, max_edge), (230, 232, 235))
                display = ImageTk.PhotoImage(placeholder)
                label = ttk.Label(panel, image=display)
                label.image = display
                label.pack()
                ttk.Label(panel, text=note, foreground="#b42318", wraplength=max_edge).pack(fill="x")
                self.photo_refs.append(display)
                return
            if view.ndim == 2:
                pil = Image.fromarray(view).convert("RGB")
                resample = Image.Resampling.NEAREST
            else:
                pil = Image.fromarray(view)
                resample = Image.Resampling.LANCZOS
            pil.thumbnail((max_edge, max_edge), resample)
            display = ImageTk.PhotoImage(pil)
            label = ttk.Label(panel, image=display)
            label.image = display
            label.pack()
            self.photo_refs.append(display)
            if note:
                ttk.Label(panel, text=note, foreground="#b42318", wraplength=max_edge).pack(fill="x")
        pre = ttk.LabelFrame(self.panel_frame, text="Mask nguồn và pre-review cục bộ — không dùng K-Means", padding=6)
        pre.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        suggestion = row.get("suggested_status", "uncertain") or "uncertain"
        summary = row.get("pre_review_summary", "Chưa có kết quả pre-review.")
        ttk.Label(pre, text=f"Gợi ý tự động: {suggestion} — {row.get('suggested_status_reason', '')}",
                  font=("Segoe UI", 10, "bold"), wraplength=1200).grid(row=0, column=0, columnspan=3, sticky="w", padx=5, pady=3)
        ttk.Label(pre, text=summary, foreground="#7a271a" if row.get("pre_review_warnings") else "#027a48",
                  wraplength=1200).grid(row=1, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        warning_codes = [code for code in row.get("pre_review_warnings", "").split(";") if code]
        warning_text = ("; ".join(WARNING_LABELS.get(code, code) for code in warning_codes)
                        if warning_codes else "Không có cảnh báo hình học/kỹ thuật.")
        ttk.Label(pre, text=f"Cảnh báo: {warning_text}", wraplength=1200).grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        ttk.Label(pre, text=(f"Tỷ lệ mask/ảnh: {row.get('mask_area_fraction') or 'không khả dụng'} | "
                             f"Tỷ lệ mask nằm trong fruit mask heuristic: {row.get('fruit_mask_overlap_fraction') or 'không khả dụng'} | "
                             f"Bbox xyxy: {row.get('mask_bbox_xyxy') or 'không khả dụng'} | "
                             f"Mask chạm biên: {row.get('touches_image_border') or 'không khả dụng'}"),
                  wraplength=1200).grid(row=3, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        ttk.Label(pre, text=row.get("vision_reason", "Vision API không được cấu hình; chỉ chạy kiểm tra cục bộ."),
                  foreground="#475467", wraplength=1200).grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=2)
        pre_images = ttk.Frame(pre)
        pre_images.grid(row=5, column=0, columnspan=3, sticky="nsew")
        pre_views = [
            ("Ảnh RGB gốc", rgb, image_error),
            ("Mask nguồn (trắng = vùng được đánh dấu)", source_mask,
             "Mask không đọc được, không binary hoặc không cùng kích thước ảnh" if source_mask is None else ""),
            ("Overlay mask nguồn (xanh = vùng mask)",
             _blend(rgb, source_mask, (40, 220, 80), alpha) if source_mask is not None else None,
             "Không thể tạo overlay do mask không hợp lệ" if source_mask is None else ""),
        ]
        for i, (title, view, note) in enumerate(pre_views):
            add_view(pre_images, i, 3, title, view, note)

        compare = ttk.LabelFrame(self.panel_frame, text="So sánh K-Means (chỉ để người review đối chiếu; không ảnh hưởng suggested_status)", padding=6)
        compare.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        compare_views = [
            ("Original + Lab K=3 — mask", lab_mask, f"Trạng thái: {row.get('lab_k3_status', '')}" if lab_mask is None else ""),
            ("Original + Lab K=3 — overlay đỏ", _blend(rgb, lab_mask, (250, 55, 45), alpha) if lab_mask is not None else None,
             "Không có mask K-Means hợp lệ" if lab_mask is None else ""),
            ("Original + HSV K=5 — mask", hsv_mask, f"Trạng thái: {row.get('hsv_k5_status', '')}" if hsv_mask is None else ""),
            ("Original + HSV K=5 — overlay đỏ", _blend(rgb, hsv_mask, (250, 55, 45), alpha) if hsv_mask is not None else None,
             "Không có mask K-Means hợp lệ" if hsv_mask is None else ""),
        ]
        for i, (title, view, note) in enumerate(compare_views):
            add_view(compare, i, 2, title, view, note)
        self.canvas.yview_moveto(0)
        self.canvas.xview_moveto(0)

    def decide(self, status: str) -> None:
        reviewer = self.reviewer.get().strip()
        notes = self.notes.get("1.0", "end").strip()
        if not reviewer:
            messagebox.showwarning("Thiếu người review", "Nhập tên/người review để lưu quyết định có thể truy vết.", parent=self.root)
            return
        if status == "approved" and not notes:
            messagebox.showwarning("Cần ghi chú xác nhận", "Ghi xác nhận bạn đã kiểm tra ý nghĩa và căn chỉnh mask trước khi approved.", parent=self.root)
            return
        prompt = {
            "approved": "Bạn đã trực tiếp kiểm tra mask nguồn căn chỉnh đúng và biểu diễn hợp lý vùng khuyết tật?",
            "rejected": "Xác nhận đánh dấu cặp này rejected?",
            "uncertain": "Xác nhận đánh dấu cặp này uncertain?",
        }[status]
        if not messagebox.askyesno("Xác nhận quyết định", prompt, parent=self.root):
            return
        try:
            set_kmeans_pair_review(self.root_path, self._current()["image_id"], status, reviewer, notes)
        except Exception as exc:
            messagebox.showerror("Không lưu được quyết định", str(exc), parent=self.root)
            return
        self._current()["review_status"] = status
        self._current()["reviewer"] = reviewer
        self._current()["review_notes"] = notes
        self._current()["reviewed_at"] = "saved"
        self._save_session()
        if self.index + 1 < len(self.rows):
            self.index += 1
        self._show_current()

    def previous(self) -> None:
        self.index = max(0, self.index - 1)
        self._show_current()

    def next(self) -> None:
        self.index = min(len(self.rows) - 1, self.index + 1)
        self._show_current()

    def evaluate(self) -> None:
        try:
            result = evaluate_reviewed_kmeans_pairs(self.root_path)
        except Exception as exc:
            messagebox.showerror("Không thể đánh giá", str(exc), parent=self.root)
            return
        messagebox.showinfo("Đánh giá đã cập nhật",
                            f"Approved: {result['approved_pairs']}\nMetric rows: {result['metric_rows']}\n"
                            f"Báo cáo: {result['report_path']}", parent=self.root)

    def close(self) -> None:
        self._save_session()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_review_gui(root: str | Path) -> None:
    KMeansReviewApp(Path(root)).run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive human review of selected Dataset C image/mask pairs")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run_review_gui(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
