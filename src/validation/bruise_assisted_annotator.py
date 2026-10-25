from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.color.preprocess import decode_image
from src.validation.bruise_assisted import (
    ASSISTED_FIELDS, ASSISTED_MANIFEST, _atomic_png, _validate_candidate,
    review_assisted_candidates,
)
from src.validation.bruise_ground_truth import _project_path, _read_csv, _write_csv, sha256_file


class AssistedBrushAnnotator:
    """Brush editor for Codex-assisted candidates; never auto-approves edits."""

    def __init__(self, root_path: str | Path, image_id: str | None = None) -> None:
        self.root = Path(root_path).resolve()
        self.manifest_path = self.root / ASSISTED_MANIFEST
        self.rows = _read_csv(self.manifest_path)
        if image_id:
            self.rows = [row for row in self.rows if row.get("image_id") == image_id]
            if not self.rows:
                raise ValueError(f"Unknown assisted image_id: {image_id}")
            if self.rows[0].get("review_status") != "pending":
                raise ValueError("Reviewed candidate is locked; explicitly return it to pending before editing")
        else:
            self.rows = [row for row in self.rows if row.get("candidate_mask_path")
                         and row.get("review_status") == "pending"]
        if not self.rows:
            raise ValueError("No pending assisted candidate masks are available")
        self.index = 0
        self.frame = None
        self.mask: np.ndarray | None = None
        self.dirty = False
        self.stroke_active = False
        self.undo_stack: list[np.ndarray] = []
        self.display_scale = 1.0
        self.zoom = 1.0
        self.tk_image: ImageTk.PhotoImage | None = None
        self.last_point: tuple[int, int] | None = None

        self.window = tk.Tk()
        sw, sh = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.title("MangoResearch | Assisted Bruise Mask Review")
        self.window.geometry(f"{min(1500, sw - 40)}x{min(980, sh - 50)}")
        self.window.minsize(980, 650)
        self.status = tk.StringVar()
        tk.Label(self.window, textvariable=self.status, anchor="w", padx=10,
                 pady=7, font=("Segoe UI", 10, "bold")).pack(fill="x")
        tk.Label(self.window,
                 text="Ảnh RGB gốc + candidate codex_assisted. Chỉnh bằng brush; lưu xong vẫn pending cho đến khi bạn duyệt.",
                 anchor="w", padx=10, fg="#444").pack(fill="x")
        body = tk.Frame(self.window)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        view = tk.Frame(body, bg="#20252b")
        view.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(view, bg="#20252b", highlightthickness=0, cursor="crosshair")
        self.xscroll = tk.Scrollbar(view, orient="horizontal", command=self.canvas.xview)
        self.yscroll = tk.Scrollbar(view, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll.grid(row=1, column=0, sticky="ew")
        view.rowconfigure(0, weight=1)
        view.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        panel = tk.Frame(body, width=250, padx=10, pady=8)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        tk.Label(panel, text="Brush", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.mode = tk.StringVar(value="add")
        tk.Radiobutton(panel, text="Thêm vùng", variable=self.mode, value="add").pack(anchor="w")
        tk.Radiobutton(panel, text="Xóa vùng", variable=self.mode, value="erase").pack(anchor="w")
        tk.Label(panel, text="Đường kính (pixel ảnh gốc)").pack(anchor="w", pady=(8, 0))
        self.brush_size = tk.IntVar(value=36)
        tk.Scale(panel, from_=4, to=240, orient="horizontal", variable=self.brush_size,
                 length=220).pack(fill="x")
        tk.Label(panel, text="Tình trạng tổn thương", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(8, 0))
        self.presence = tk.StringVar(value="not_reviewed")
        for label, value in (("Chưa xác định", "not_reviewed"), ("Có tổn thương rõ", "present"),
                             ("Không thấy tổn thương rõ", "absent"), ("Chưa chắc chắn", "uncertain")):
            tk.Radiobutton(panel, text=label, variable=self.presence, value=value, anchor="w").pack(fill="x")
        tk.Label(panel, text="Người duyệt / initials").pack(anchor="w", pady=(10, 0))
        self.reviewer = tk.StringVar()
        tk.Entry(panel, textvariable=self.reviewer).pack(fill="x")
        tk.Label(panel, text="Ghi chú").pack(anchor="w", pady=(8, 0))
        self.notes = tk.Text(panel, height=5, wrap="word")
        self.notes.pack(fill="x")
        self._button(panel, "Undo nét vừa vẽ", self._undo).pack(fill="x", pady=(8, 2))
        self._button(panel, "Xóa toàn bộ mask", self._clear_mask).pack(fill="x", pady=2)
        self._button(panel, "Lưu candidate (pending)", self._save_candidate, "#d8f3e6").pack(fill="x", pady=2)
        self._button(panel, "Đánh dấu uncertain", self._uncertain).pack(fill="x", pady=2)
        self._button(panel, "Duyệt candidate", self._approve, "#bfe7cd").pack(fill="x", pady=2)
        self._button(panel, "Reject candidate", self._reject).pack(fill="x", pady=2)
        footer = tk.Frame(self.window, padx=8, pady=6)
        footer.pack(fill="x")
        self._button(footer, "← Trước", self._previous).pack(side="left", padx=3)
        self._button(footer, "Tiếp →", self._next).pack(side="left", padx=3)
        self._button(footer, "Zoom −", lambda: self._zoom(0.8)).pack(side="left", padx=(16, 3))
        self._button(footer, "Zoom +", lambda: self._zoom(1.25)).pack(side="left", padx=3)
        self._button(footer, "Vừa khung", self._fit).pack(side="left", padx=3)
        self._button(footer, "Lưu nháp", self._save_draft).pack(side="right", padx=3)
        self.window.bind("<Left>", self._previous)
        self.window.bind("<Right>", self._next)
        self.window.bind("u", self._undo)
        self.window.bind("<Escape>", self._close)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._load_current()

    @staticmethod
    def _button(parent, label: str, command, background: str = "#eeeeee"):
        return tk.Button(parent, text=label, command=command, padx=7, pady=5,
                         background=background, font=("Segoe UI", 9))

    def _row(self) -> dict[str, str]:
        return self.rows[self.index]

    def _draft_path(self) -> Path:
        return self.root / "data/annotations/bruise_gt_assisted/drafts" / f"{self._row()['image_id']}.png"

    def _load_current(self) -> None:
        row = self._row()
        source_path = _project_path(self.root, row["source_image_path"])
        if sha256_file(source_path) != row["source_sha256"]:
            raise ValueError(f"Source checksum mismatch for {row['image_id']}")
        self.frame = decode_image(source_path)
        mask_path = _project_path(self.root, row["candidate_mask_path"])
        if sha256_file(mask_path) != row["candidate_mask_sha256"]:
            raise ValueError(f"Candidate checksum mismatch for {row['image_id']}; regenerate or restore it first")
        original = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if original is None or original.shape != self.frame.rgb.shape[:2] or not set(np.unique(original)).issubset({0, 255}):
            raise ValueError(f"Candidate mask dimensions/values invalid for {row['image_id']}")
        draft_path = self._draft_path()
        if draft_path.is_file():
            saved = cv2.imread(str(draft_path), cv2.IMREAD_GRAYSCALE)
            if saved is not None and saved.shape == original.shape and set(np.unique(saved)).issubset({0, 255}):
                original = saved
        self.mask = original
        self.dirty = False
        self.undo_stack.clear()
        self.presence.set(row.get("lesion_presence") or "not_reviewed")
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", row.get("visual_review_note", ""))
        self._render()
        self._refresh_status()

    def _refresh_status(self) -> None:
        row = self._row()
        self.status.set(f"{self.index + 1}/{len(self.rows)} · {row['image_id']} · {Path(row['source_image_path']).name} · "
                        f"{row.get('annotation_status', '')} / {row.get('review_status', '')} / "
                        f"presence={row.get('lesion_presence', 'not_reviewed')} · mask {self.frame.rgb.shape[1]}×{self.frame.rgb.shape[0]}")

    def _render(self) -> None:
        rgb = self.frame.rgb.astype(np.float32)
        active = self.mask > 0
        rgb[active] = 0.48 * rgb[active] + 0.52 * np.array([255, 35, 145], dtype=np.float32)
        shown_rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        self.display_scale = min(1100 / shown_rgb.shape[1], 690 / shown_rgb.shape[0], 1.0) * self.zoom
        size = (max(1, round(shown_rgb.shape[1] * self.display_scale)),
                max(1, round(shown_rgb.shape[0] * self.display_scale)))
        shown = Image.fromarray(shown_rgb).resize(size, Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, *size))

    def _point(self, event) -> tuple[int, int]:
        h, w = self.mask.shape
        x = round(self.canvas.canvasx(event.x) / self.display_scale)
        y = round(self.canvas.canvasy(event.y) / self.display_scale)
        return min(max(x, 0), w - 1), min(max(y, 0), h - 1)

    def _paint_to(self, point: tuple[int, int]) -> None:
        if self.mask is None:
            return
        color = 255 if self.mode.get() == "add" else 0
        radius = max(1, int(self.brush_size.get()) // 2)
        if self.last_point is not None:
            cv2.line(self.mask, self.last_point, point, color, thickness=radius * 2,
                     lineType=cv2.LINE_8)
        cv2.circle(self.mask, point, radius, color, thickness=-1, lineType=cv2.LINE_8)
        self.last_point = point
        self.dirty = True
        self._render()

    def _press(self, event) -> None:
        self.undo_stack.append(self.mask.copy())
        self.undo_stack = self.undo_stack[-40:]
        self.stroke_active = True
        self.last_point = None
        self._paint_to(self._point(event))

    def _drag(self, event) -> None:
        if self.stroke_active:
            self._paint_to(self._point(event))

    def _release(self, _event) -> None:
        self.stroke_active = False
        self.last_point = None

    def _undo(self, _event=None) -> None:
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self.dirty = True
            self._render()

    def _clear_mask(self) -> None:
        if self.mask is None or not np.any(self.mask):
            return
        if not messagebox.askyesno("Xóa mask", "Xóa toàn bộ vùng đang tô trên candidate này?"):
            return
        self.undo_stack.append(self.mask.copy())
        self.undo_stack = self.undo_stack[-40:]
        self.mask.fill(0)
        self.dirty = True
        self._render()

    def _write_manifest(self) -> None:
        _write_csv(self.manifest_path, self.rows, ASSISTED_FIELDS)

    def _save_draft(self, _event=None) -> None:
        if self.mask is None:
            return
        _atomic_png(self._draft_path(), self.mask)
        self._row()["visual_review_note"] = self.notes.get("1.0", "end").strip()
        self._row()["review_status"] = "pending"
        self._write_manifest()
        self.dirty = False

    def _save_candidate(self) -> bool:
        if self.mask is None:
            return False
        row = self._row()
        if row.get("review_status") == "approved":
            messagebox.showerror("Candidate đã duyệt", "Hãy đưa candidate về pending bằng quyết định tường minh trước khi sửa.")
            return False
        candidate = _project_path(self.root, row["candidate_mask_path"])
        validation = _validate_candidate(self.frame.rgb, self.mask)
        if not validation["passed"]:
            messagebox.showerror("Mask không hợp lệ", validation["note"])
            return False
        presence = self.presence.get()
        positives = int(np.count_nonzero(self.mask))
        if presence == "present" and positives == 0:
            messagebox.showerror("Mask rỗng", "Đã chọn có tổn thương nhưng mask đang rỗng.")
            return False
        if presence == "absent" and positives:
            messagebox.showerror("Mask không rỗng", "Đã chọn không thấy tổn thương; xóa mask trước khi lưu.")
            return False
        _atomic_png(candidate, self.mask)
        row["candidate_mask_sha256"] = sha256_file(candidate)
        row["mask_width"], row["mask_height"] = str(self.mask.shape[1]), str(self.mask.shape[0])
        row["mask_values"] = "0/255"
        row["technical_validation_status"] = validation["status"]
        row["technical_validation_note"] = validation["note"]
        row["technical_validation_warnings"] = json.dumps(validation["warnings"], ensure_ascii=False)
        row["outside_fruit_fraction"] = validation["outside_fraction"]
        positives = int(np.count_nonzero(self.mask))
        components = cv2.connectedComponents((self.mask > 0).astype(np.uint8), connectivity=8)[0] - 1
        row["candidate_area_fraction"] = f"{positives / self.mask.size:.8f}"
        row["component_count"] = str(components)
        row["annotation_origin"] = "codex_assisted"
        row["review_status"] = "pending"
        row["annotation_status"] = "edited_pending_review"
        row["lesion_presence"] = presence
        row["uncertain"] = "false"
        row["visual_review_note"] = self.notes.get("1.0", "end").strip()
        row["edited_by"] = self.reviewer.get().strip()
        row["edited_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        row["reviewer"], row["reviewed_at"] = "", ""
        self._write_manifest()
        draft_path = self._draft_path()
        if draft_path.exists():
            draft_path.unlink()
        self.dirty = False
        self._refresh_status()
        messagebox.showinfo("Đã lưu candidate", "Mask ứng viên đã cập nhật; semantic review vẫn pending.")
        return True

    def _set_review(self, status: str) -> None:
        reviewer = self.reviewer.get().strip()
        note = self.notes.get("1.0", "end").strip()
        if not reviewer:
            messagebox.showerror("Thiếu người duyệt", "Nhập tên hoặc initials trước khi duyệt trạng thái.")
            return
        if self.dirty:
            if not self._save_candidate():
                return
        if status in {"uncertain", "rejected"} and not note:
            messagebox.showerror("Thiếu ghi chú", "Ghi ngắn gọn lý do uncertain/reject.")
            return
        if status == "approved":
            presence = self.presence.get()
            positives = int(np.count_nonzero(self.mask))
            if presence not in {"present", "absent"}:
                messagebox.showerror("Chưa xác định", "Chọn có tổn thương hoặc không thấy tổn thương trước khi duyệt.")
                return
            if presence == "present" and positives == 0:
                messagebox.showerror("Mask rỗng", "Tổn thương được đánh dấu present cần mask không rỗng.")
                return
            if presence == "absent" and positives:
                messagebox.showerror("Mask không rỗng", "Chọn absent chỉ khi candidate mask rỗng.")
                return
        if not messagebox.askyesno("Xác nhận review", f"Đánh dấu candidate là {status}? Mask sẽ vẫn nằm ở bộ assisted."):
            return
        row = self._row()
        try:
            review_assisted_candidates(
                self.root, [row["image_id"]], status, reviewer, note,
                lesion_presence=self.presence.get() if status == "approved" else None,
                refresh_contact_sheets=False,
                confirmation_phrase=f"APPROVE 1 ASSISTED MASKS" if status == "approved" else None,
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không thể ghi trạng thái", str(exc))
            return
        messagebox.showinfo("Đã ghi trạng thái", f"{status} được ghi trong manifest assisted. Ground-truth thủ công chưa bị thay đổi.")
        self.rows.pop(self.index)
        if not self.rows:
            self.window.destroy()
            return
        self.index = min(self.index, len(self.rows) - 1)
        self._load_current()

    def _approve(self) -> None:
        self._set_review("approved")

    def _uncertain(self) -> None:
        self._set_review("uncertain")

    def _reject(self) -> None:
        self._set_review("rejected")

    def _navigate(self, delta: int) -> None:
        if self.dirty:
            self._save_draft()
        self.index = (self.index + delta) % len(self.rows)
        self._load_current()

    def _previous(self, _event=None) -> None:
        self._navigate(-1)

    def _next(self, _event=None) -> None:
        self._navigate(1)

    def _zoom(self, factor: float) -> None:
        self.zoom = min(5.0, max(0.6, self.zoom * factor))
        self._render()

    def _fit(self) -> None:
        self.zoom = 1.0
        self._render()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _close(self, _event=None) -> None:
        if self.dirty:
            self._save_draft()
        self.window.destroy()

    def run(self) -> None:
        self.window.mainloop()


def run_assisted_annotator(root: str | Path, image_id: str | None = None) -> None:
    AssistedBrushAnnotator(root, image_id).run()
