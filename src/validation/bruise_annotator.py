from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.color.preprocess import decode_image
from src.validation.bruise_ground_truth import (
    ANNOTATION_ROOT, DRAFT_DIR, MANIFEST_PATH, _manifest_rows, _project_path,
    _replace_row, _save_candidate, approve_masks, rasterize_polygons, review_status,
)


def _mask_to_polygons(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[list[tuple[int, int]]] = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, max(1.0, perimeter * 0.0015), True).reshape(-1, 2)
        if len(polygon) >= 3:
            result.append([(int(x), int(y)) for x, y in polygon])
    result.sort(key=lambda points: (min(y for _, y in points), min(x for x, _ in points)))
    return result


class BruiseAnnotator:
    """RGB-only, resumable polygon editor for independent bruise annotations."""

    def __init__(self, root_path: str | Path, image_id: str | None = None) -> None:
        self.project_root = Path(root_path).resolve()
        self.manifest_path = self.project_root / MANIFEST_PATH
        self.draft_dir = self.project_root / DRAFT_DIR
        self.session_path = self.draft_dir / "session.json"
        self.rows = self._load_open_rows(image_id)
        if not self.rows:
            raise ValueError("No pending, draft, candidate, or uncertain annotations are available")
        self.index = self._resume_index(image_id)
        self.polygons: list[list[tuple[int, int]]] = []
        self.polygon_types: list[str] = []
        self.in_progress: list[tuple[int, int]] | None = None
        self.in_progress_type: str | None = None
        self.selected_polygon = 0
        self.active_vertex: tuple[int, int] | None = None
        self.drag_snapshot_taken = False
        self.undo_stack: list[tuple[list[list[tuple[int, int]]], list[str],
                                    list[tuple[int, int]] | None, str | None]] = []
        self.dirty = False
        self.base_scale = 1.0
        self.zoom = 1.0
        self.display_scale = 1.0
        self.tk_image: ImageTk.PhotoImage | None = None
        self.frame = None

        self.window = tk.Tk()
        screen_w, screen_h = self.window.winfo_screenwidth(), self.window.winfo_screenheight()
        self.window.title("MangoResearch | Independent Bruise Ground Truth")
        self.window.geometry(f"{min(1500, screen_w - 40)}x{min(980, screen_h - 50)}")
        self.window.minsize(980, 650)

        header = tk.Frame(self.window, padx=12, pady=8)
        header.pack(fill="x")
        self.status = tk.StringVar()
        tk.Label(header, textvariable=self.status, anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x")
        tk.Label(header,
                 text="Chỉ dùng ảnh RGB gốc và polygon bạn đang vẽ. Không hiển thị mask Dataset C hoặc kết quả K-Means.",
                 anchor="w", fg="#444", font=("Segoe UI", 9)).pack(fill="x", pady=(4, 0))

        body = tk.Frame(self.window)
        body.pack(fill="both", expand=True, padx=8, pady=4)
        canvas_frame = tk.Frame(body, background="#20252b")
        canvas_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_frame, background="#20252b", cursor="crosshair", highlightthickness=0)
        self.xscroll = tk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.yscroll = tk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.xscroll.set, yscrollcommand=self.yscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.yscroll.grid(row=0, column=1, sticky="ns")
        self.xscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)

        panel = tk.Frame(body, width=270, padx=10, pady=8)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        tk.Label(panel, text="Các polygon", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.polygon_list = tk.Listbox(panel, height=8, exportselection=False)
        self.polygon_list.pack(fill="x", pady=(4, 6))
        self.polygon_list.bind("<<ListboxSelect>>", self.on_select_polygon)
        self._button(panel, "Thêm polygon", self.begin_polygon).pack(fill="x", pady=2)
        self._button(panel, "Hoàn tất polygon", self.finish_polygon, "#dcecff").pack(fill="x", pady=2)
        self._button(panel, "Xóa polygon đã chọn", self.delete_polygon).pack(fill="x", pady=2)
        self._button(panel, "Xóa tất cả polygon", self.clear_polygons).pack(fill="x", pady=2)
        self._button(panel, "Undo", self.undo).pack(fill="x", pady=(2, 10))

        tk.Label(panel, text="Loại tổn thương của polygon", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(2, 0))
        self.injury_type = tk.StringVar(value="bruise_spot")
        self.injury_type_box = ttk.Combobox(
            panel, textvariable=self.injury_type, state="readonly",
            values=("bruise_spot", "peel_damage", "scar", "tear", "unknown"),
        )
        self.injury_type_box.pack(fill="x", pady=(3, 8))
        self.injury_type_box.bind("<<ComboboxSelected>>", self.on_change_injury_type)

        tk.Label(panel, text="Tình trạng quan sát", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.presence = tk.StringVar(value="present")
        for label, value in (("Có tổn thương rõ", "present"), ("Không thấy tổn thương rõ", "absent"),
                             ("Chưa chắc chắn", "uncertain")):
            tk.Radiobutton(panel, text=label, variable=self.presence, value=value, anchor="w").pack(fill="x")
        tk.Label(panel, text="Người chú giải / initials").pack(anchor="w", pady=(10, 0))
        self.reviewer = tk.StringVar()
        tk.Entry(panel, textvariable=self.reviewer).pack(fill="x")
        tk.Label(panel, text="Ghi chú chú giải").pack(anchor="w", pady=(7, 0))
        self.note_text = tk.Text(panel, height=4, wrap="word")
        self.note_text.pack(fill="x")
        self._button(panel, "Lưu ứng viên mask", self.save_candidate, "#d8f3e6").pack(fill="x", pady=(9, 2))
        self._button(panel, "Duyệt mask hiện tại", self.approve_current, "#bfe7cd").pack(fill="x", pady=2)
        self._button(panel, "Đánh dấu uncertain", self.mark_uncertain).pack(fill="x", pady=2)
        self._button(panel, "Reject ảnh", self.reject_current).pack(fill="x", pady=2)

        controls = tk.Frame(self.window, padx=8, pady=7)
        controls.pack(fill="x")
        self._button(controls, "← Trước", self.previous).pack(side="left", padx=3)
        self._button(controls, "Tiếp →", self.next).pack(side="left", padx=3)
        self._button(controls, "Zoom −", self.zoom_out).pack(side="left", padx=(18, 3))
        self._button(controls, "Zoom +", self.zoom_in).pack(side="left", padx=3)
        self._button(controls, "Vừa khung", self.fit_view).pack(side="left", padx=3)
        self._button(controls, "Lưu nháp", self.save_draft).pack(side="right", padx=3)
        footer = tk.Label(self.window,
                          text="Bấm để thêm đỉnh vào polygon mới; kéo đỉnh để sửa; bấm cạnh để chèn đỉnh; chuột phải lên đỉnh để xóa. Dùng thanh cuộn hoặc nhấn giữ nút giữa để kéo ảnh. ←/→: chuyển ảnh · +/-: zoom · U: undo · Esc: lưu nháp và đóng.",
                          anchor="w", padx=12, pady=4, fg="#555", font=("Segoe UI", 9))
        footer.pack(fill="x")
        self.window.bind("<Left>", self.previous)
        self.window.bind("<Right>", self.next)
        self.window.bind("<Escape>", self.close)
        self.window.bind("u", self.undo)
        self.window.bind("+", self.zoom_in)
        self.window.bind("-", self.zoom_out)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.show_current()

    @staticmethod
    def _button(parent: tk.Misc, text: str, command, background: str = "#eeeeee") -> tk.Button:
        return tk.Button(parent, text=text, command=command, padx=7, pady=5,
                         font=("Segoe UI", 9), background=background, activebackground="#dcecff")

    def _load_open_rows(self, image_id: str | None) -> list[dict[str, str]]:
        rows = _manifest_rows(self.project_root)
        if image_id:
            selected = [row for row in rows if row.get("image_id") == image_id]
            if not selected:
                raise ValueError(f"Unknown image_id: {image_id}")
            if selected[0].get("approval_status") == "approved":
                raise ValueError("Approved masks are locked; have the user return the row to pending before editing")
            return selected
        return [row for row in rows if row.get("approval_status") not in {"approved", "rejected"}]

    def _resume_index(self, requested_image_id: str | None) -> int:
        if requested_image_id:
            return 0
        if self.session_path.is_file():
            try:
                current_id = json.loads(self.session_path.read_text(encoding="utf-8")).get("current_image_id")
                for index, row in enumerate(self.rows):
                    if row.get("image_id") == current_id:
                        return index
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        for index, row in enumerate(self.rows):
            if row.get("workflow_status") in {"draft", "candidate", "uncertain"}:
                return index
        return 0

    def _save_session(self) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        current = self.rows[self.index]["image_id"]
        temporary = self.session_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"current_image_id": current,
                                         "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.session_path)

    def _note(self) -> str:
        return self.note_text.get("1.0", "end").strip()

    def _write_draft(self) -> None:
        row = self.rows[self.index]
        data = {
            "image_id": row["image_id"], "source_image_path": row["source_image_path"],
            "source_sha256": row["source_sha256"], "width": int(row["width"]),
            "height": int(row["height"]), "polygons": self.polygons,
            "polygon_types": self.polygon_types,
            "in_progress": self.in_progress,
            "in_progress_type": self.in_progress_type,
            "lesion_presence": self.presence.get(), "draft_status": "draft",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "annotation_note": self._note(),
        }
        target = self.draft_dir / f"{row['image_id']}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)
        _replace_row(self.project_root, {"image_id": row["image_id"], "workflow_status": "draft",
                                         "approval_status": "pending", "semantic_review_notes": self._note()})
        self.rows[self.index]["workflow_status"] = "draft"
        self.dirty = False
        self._save_session()
        self._refresh_status()

    def _load_draft_or_candidate(self, row: dict[str, str]) -> list[list[tuple[int, int]]]:
        draft_path = self.draft_dir / f"{row['image_id']}.json"
        source_path = _project_path(self.project_root, row["source_image_path"])
        if draft_path.is_file():
            try:
                saved = json.loads(draft_path.read_text(encoding="utf-8"))
                if (saved.get("image_id") == row["image_id"]
                        and saved.get("source_sha256") == row["source_sha256"]
                        and int(saved.get("width", -1)) == int(row["width"])
                        and int(saved.get("height", -1)) == int(row["height"])):
                    self.presence.set(saved.get("lesion_presence", "present"))
                    self.note_text.delete("1.0", "end")
                    self.note_text.insert("1.0", saved.get("annotation_note", ""))
                    saved_polygons = saved.get("polygons", [])
                    saved_types = saved.get("polygon_types", [])
                    self.polygon_types = [str(value) for value in saved_types[:len(saved_polygons)]]
                    self.polygon_types.extend(["unknown"] * (len(saved_polygons) - len(self.polygon_types)))
                    progress = saved.get("in_progress")
                    self.in_progress = ([(int(x), int(y)) for x, y in progress]
                                        if isinstance(progress, list) else None)
                    self.in_progress_type = saved.get("in_progress_type") if self.in_progress is not None else None
                    return [[(int(x), int(y)) for x, y in polygon] for polygon in saved_polygons]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        if row.get("mask_path"):
            mask_path = _project_path(self.project_root, row["mask_path"])
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
            if mask is not None and mask.shape == (int(row["height"]), int(row["width"])):
                self.presence.set(row.get("lesion_presence") or "present")
                self.note_text.delete("1.0", "end")
                self.note_text.insert("1.0", row.get("semantic_review_notes", ""))
                polygons = _mask_to_polygons(mask)
                self.polygon_types = ["unknown"] * len(polygons)
                return polygons
        self.presence.set("present")
        self.note_text.delete("1.0", "end")
        return []

    def show_current(self) -> None:
        row = self.rows[self.index]
        self.frame = decode_image(_project_path(self.project_root, row["source_image_path"]))
        self.polygon_types = []
        self.in_progress = None
        self.in_progress_type = None
        self.polygons = self._load_draft_or_candidate(row)
        self.selected_polygon = min(self.selected_polygon, max(0, len(self.polygons) - 1))
        self.undo_stack.clear()
        self.active_vertex = None
        self.dirty = False
        rgb = self.frame.rgb
        self.base_scale = min(1100 / rgb.shape[1], 690 / rgb.shape[0], 1.0)
        self.zoom = 1.0
        self._redraw()
        self._refresh_polygon_list()
        self._refresh_status()
        self._save_session()

    def _refresh_status(self) -> None:
        row = self.rows[self.index]
        total = len(_manifest_rows(self.project_root))
        self.status.set(f"{self.index + 1}/{len(self.rows)}  |  {total} candidates  |  {row['dataset']} / {Path(row['source_image_path']).name}  |  "
                        f"{row['image_id']}  |  {row['width']}×{row['height']}  |  "
                        f"{row.get('workflow_status', 'pending')} / {row.get('approval_status', 'pending')}")

    def _redraw(self) -> None:
        if self.frame is None:
            return
        self.canvas.delete("all")
        self.display_scale = self.base_scale * self.zoom
        height, width = self.frame.rgb.shape[:2]
        shown_w, shown_h = max(1, round(width * self.display_scale)), max(1, round(height * self.display_scale))
        shown = Image.fromarray(self.frame.rgb).resize((shown_w, shown_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(shown)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.configure(scrollregion=(0, 0, shown_w, shown_h))
        for index, polygon in enumerate(self.polygons):
            if len(polygon) < 2:
                continue
            coords = [coord for point in polygon for coord in (point[0] * self.display_scale,
                                                                 point[1] * self.display_scale)]
            self.canvas.create_polygon(*coords, fill="#35d06f", stipple="gray25",
                                       outline="#00bf56" if index == self.selected_polygon else "#ffe06b",
                                       width=2 if index == self.selected_polygon else 1)
            for x, y in polygon:
                radius = 5
                px, py = x * self.display_scale, y * self.display_scale
                self.canvas.create_oval(px - radius, py - radius, px + radius, py + radius,
                                        fill="#00d65c", outline="white", width=1)
        if self.in_progress:
            coords = [coord for point in self.in_progress for coord in (point[0] * self.display_scale,
                                                                          point[1] * self.display_scale)]
            if len(coords) >= 4:
                self.canvas.create_line(*coords, fill="#00eaff", width=2, dash=(4, 2))
            for x, y in self.in_progress:
                px, py = x * self.display_scale, y * self.display_scale
                self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#00eaff", outline="white")

    def _refresh_polygon_list(self) -> None:
        self.polygon_list.delete(0, "end")
        for index, polygon in enumerate(self.polygons, start=1):
            injury_type = self.polygon_types[index - 1] if index - 1 < len(self.polygon_types) else "unknown"
            self.polygon_list.insert("end", f"Tổn thương {index} · {injury_type} · {len(polygon)} đỉnh")
        if self.polygons:
            self.selected_polygon = min(self.selected_polygon, len(self.polygons) - 1)
            self.polygon_list.selection_set(self.selected_polygon)

    def _snapshot(self) -> None:
        self.undo_stack.append(([list(poly) for poly in self.polygons], list(self.polygon_types),
                                list(self.in_progress) if self.in_progress is not None else None,
                                self.in_progress_type))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self, _event=None) -> None:
        if not self.undo_stack:
            return
        self.polygons, self.polygon_types, self.in_progress, self.in_progress_type = self.undo_stack.pop()
        self.dirty = True
        self._redraw()
        self._refresh_polygon_list()
        self._write_draft()

    def begin_polygon(self) -> None:
        if self.in_progress is not None:
            messagebox.showinfo("Polygon đang vẽ", "Hoàn tất hoặc hủy polygon hiện tại trước.")
            return
        self._snapshot()
        self.in_progress = []
        self.in_progress_type = self.injury_type.get()
        self.dirty = True
        self._redraw()

    def finish_polygon(self) -> None:
        if self.in_progress is None or len(self.in_progress) < 3:
            messagebox.showinfo("Chưa đủ đỉnh", "Một polygon cần ít nhất 3 đỉnh.")
            return
        self._snapshot()
        self.polygons.append(self.in_progress)
        self.polygon_types.append(self.in_progress_type or "unknown")
        self.selected_polygon = len(self.polygons) - 1
        self.in_progress = None
        self.in_progress_type = None
        self.dirty = True
        self._redraw()
        self._refresh_polygon_list()
        self._write_draft()

    def delete_polygon(self) -> None:
        if self.in_progress is not None:
            self.in_progress = None
            self.in_progress_type = None
            self.dirty = True
            self._redraw()
            return
        if not self.polygons:
            return
        self._snapshot()
        del self.polygons[self.selected_polygon]
        del self.polygon_types[self.selected_polygon]
        self.selected_polygon = min(self.selected_polygon, max(0, len(self.polygons) - 1))
        self.dirty = True
        self._redraw()
        self._refresh_polygon_list()
        self._write_draft()

    def clear_polygons(self) -> None:
        if not self.polygons and self.in_progress is None:
            return
        if not messagebox.askyesno("Xóa tất cả", "Xóa tất cả polygon trên ảnh này?"):
            return
        self._snapshot()
        self.polygons = []
        self.polygon_types = []
        self.in_progress = None
        self.in_progress_type = None
        self.selected_polygon = 0
        self.dirty = True
        self._redraw()
        self._refresh_polygon_list()
        self._write_draft()

    def on_select_polygon(self, _event=None) -> None:
        selection = self.polygon_list.curselection()
        if selection:
            self.selected_polygon = int(selection[0])
            self.injury_type.set(self.polygon_types[self.selected_polygon])
            self._redraw()

    def on_change_injury_type(self, _event=None) -> None:
        chosen = self.injury_type.get()
        if self.in_progress is not None:
            self.in_progress_type = chosen
        elif self.polygons:
            self._snapshot()
            self.polygon_types[self.selected_polygon] = chosen
            self.dirty = True
            self._refresh_polygon_list()
            self._write_draft()

    def _canvas_point(self, event: tk.Event) -> tuple[int, int]:
        x = self.canvas.canvasx(event.x) / self.display_scale
        y = self.canvas.canvasy(event.y) / self.display_scale
        height, width = self.frame.rgb.shape[:2]
        return min(max(round(x), 0), width - 1), min(max(round(y), 0), height - 1)

    @staticmethod
    def _point_segment_distance(point, a, b) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom == 0 else min(max(((px - ax) * dx + (py - ay) * dy) / denom, 0.0), 1.0)
        return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))

    def on_press(self, event: tk.Event) -> None:
        point = self._canvas_point(event)
        if self.in_progress is not None:
            self._snapshot()
            self.in_progress.append(point)
            self.dirty = True
            self._redraw()
            return
        if not self.polygons:
            messagebox.showinfo("Tạo polygon", "Nhấn ‘Thêm polygon’ trước khi đặt đỉnh.")
            return
        radius = 12 / self.display_scale
        polygon = self.polygons[self.selected_polygon]
        distances = [np.hypot(point[0] - x, point[1] - y) for x, y in polygon]
        nearest = int(np.argmin(distances)) if distances else -1
        if nearest >= 0 and distances[nearest] <= radius:
            self._snapshot()
            self.active_vertex = (self.selected_polygon, nearest)
            self.drag_snapshot_taken = True
            return
        if len(polygon) >= 2:
            edges = [(self._point_segment_distance(point, polygon[index], polygon[(index + 1) % len(polygon)]), index)
                     for index in range(len(polygon))]
            distance, edge_index = min(edges)
            if distance <= radius:
                self._snapshot()
                self.polygons[self.selected_polygon].insert(edge_index + 1, point)
                self.active_vertex = (self.selected_polygon, edge_index + 1)
                self.dirty = True
                self._redraw()
                return

    def on_drag(self, event: tk.Event) -> None:
        if self.active_vertex is None:
            return
        polygon_index, vertex_index = self.active_vertex
        self.polygons[polygon_index][vertex_index] = self._canvas_point(event)
        self.dirty = True
        self._redraw()

    def on_release(self, _event: tk.Event) -> None:
        self.active_vertex = None
        self.drag_snapshot_taken = False
        if self.dirty:
            self._write_draft()

    def on_right_click(self, event: tk.Event) -> None:
        if self.in_progress is not None:
            if self.in_progress:
                self._snapshot()
                self.in_progress.pop()
                self.dirty = True
                self._redraw()
            return
        if not self.polygons:
            return
        point = self._canvas_point(event)
        polygon = self.polygons[self.selected_polygon]
        distances = [np.hypot(point[0] - x, point[1] - y) for x, y in polygon]
        if not distances:
            return
        nearest = int(np.argmin(distances))
        if distances[nearest] * self.display_scale > 12:
            return
        self._snapshot()
        polygon.pop(nearest)
        if len(polygon) < 3:
            del self.polygons[self.selected_polygon]
            del self.polygon_types[self.selected_polygon]
            self.selected_polygon = min(self.selected_polygon, max(0, len(self.polygons) - 1))
        self.dirty = True
        self._redraw()
        self._refresh_polygon_list()
        self._write_draft()

    def on_pan_start(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_drag(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _change_zoom(self, factor: float) -> None:
        self.zoom = min(6.0, max(0.6, self.zoom * factor))
        self._redraw()

    def zoom_in(self, _event=None) -> None:
        self._change_zoom(1.25)

    def zoom_out(self, _event=None) -> None:
        self._change_zoom(0.8)

    def fit_view(self, _event=None) -> None:
        self.zoom = 1.0
        self._redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def save_draft(self, _event=None) -> None:
        self._write_draft()

    def save_candidate(self) -> None:
        row = self.rows[self.index]
        presence = self.presence.get()
        if self.in_progress is not None:
            messagebox.showerror("Polygon chưa hoàn tất", "Hoàn tất polygon đang vẽ trước khi lưu.")
            return
        if presence == "absent" and self.polygons:
            messagebox.showerror("Mask không rỗng", "Chọn ‘Có tổn thương rõ’ hoặc xóa polygon trước khi ghi nhận không có tổn thương.")
            return
        if presence == "present" and not self.polygons:
            messagebox.showerror("Mask rỗng", "Vẽ ít nhất một polygon hoặc chọn ‘Không thấy tổn thương rõ’ / ‘Chưa chắc chắn’.")
            return
        mask = rasterize_polygons(self.frame.rgb.shape[:2], self.polygons)
        try:
            saved = _save_candidate(self.project_root, row["image_id"], mask, presence,
                                    self._note(), self.polygon_types)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không thể lưu ứng viên", str(exc))
            return
        self.rows[self.index].update({"workflow_status": saved["workflow_status"],
                                      "approval_status": "pending", "mask_path": saved["mask_path"],
                                      "mask_sha256": saved["mask_sha256"], "lesion_presence": presence})
        draft = {
            "image_id": row["image_id"], "source_image_path": row["source_image_path"],
            "source_sha256": row["source_sha256"], "width": int(row["width"]),
            "height": int(row["height"]), "polygons": self.polygons,
            "polygon_types": self.polygon_types, "in_progress": None, "in_progress_type": None,
            "lesion_presence": presence, "draft_status": "candidate",
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "annotation_note": self._note(),
        }
        draft_path = self.draft_dir / f"{row['image_id']}.json"
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = draft_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(draft_path)
        self.dirty = False
        self._refresh_status()
        messagebox.showinfo("Ứng viên đã lưu", "Mask được lưu và kiểm tra kỹ thuật. Semantic approval vẫn đang chờ.")

    def approve_current(self) -> None:
        row = self.rows[self.index]
        reviewer = self.reviewer.get().strip()
        if not reviewer:
            messagebox.showerror("Thiếu người duyệt", "Nhập tên hoặc initials của người chú giải trước khi duyệt.")
            return
        if self.dirty or row.get("workflow_status") not in {"candidate", "uncertain"}:
            self.save_candidate()
            row = self.rows[self.index]
            if row.get("workflow_status") not in {"candidate", "uncertain"}:
                return
        if row.get("lesion_presence") == "uncertain":
            messagebox.showerror("Cần xử lý uncertain", "Giải quyết trạng thái uncertain trước khi duyệt.")
            return
        if not messagebox.askyesno("Duyệt ground truth", "Bạn đã kiểm tra vị trí tổn thương và mask độc lập trên ảnh RGB gốc? Mask sẽ được đưa vào metric."):
            return
        try:
            result = approve_masks(self.project_root, [row["image_id"]], reviewer, self._note())
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không thể duyệt", str(exc))
            return
        self._finish_reviewed_row("approved")
        messagebox.showinfo("Đã duyệt", f"Đã duyệt mask của {result['image_ids'][0]}.")

    def mark_uncertain(self) -> None:
        row = self.rows[self.index]
        reviewer = self.reviewer.get().strip()
        notes = self._note() or simpledialog.askstring("Lý do uncertain", "Điểm nào chưa thể xác định rõ?") or ""
        if not reviewer or not notes.strip():
            messagebox.showerror("Thiếu thông tin", "Nhập người chú giải và ghi chú ngắn cho trạng thái uncertain.")
            return
        if self.dirty:
            self._write_draft()
        try:
            review_status(self.project_root, [row["image_id"]], "uncertain", reviewer, notes)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không thể lưu trạng thái", str(exc))
            return
        self._finish_reviewed_row("uncertain")

    def reject_current(self) -> None:
        row = self.rows[self.index]
        reviewer = self.reviewer.get().strip()
        notes = simpledialog.askstring("Reject ảnh", "Ghi lý do loại ảnh (tổn thương mơ hồ, che khuất, ảnh không phù hợp):") or ""
        if not reviewer or not notes.strip():
            messagebox.showerror("Thiếu thông tin", "Nhập người chú giải và lý do reject.")
            return
        try:
            review_status(self.project_root, [row["image_id"]], "rejected", reviewer, notes)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Không thể lưu trạng thái", str(exc))
            return
        self._finish_reviewed_row("rejected")

    def _finish_reviewed_row(self, status: str) -> None:
        if len(self.rows) == 1:
            messagebox.showinfo("Hoàn tất", f"Ảnh được đánh dấu {status}. Đóng và mở lại editor để tiếp tục.")
            return
        self.rows.pop(self.index)
        if self.index >= len(self.rows):
            self.index = len(self.rows) - 1
        self.show_current()

    def _navigate(self, delta: int) -> None:
        if self.dirty:
            self._write_draft()
        if self.in_progress is not None:
            self.in_progress = None
        self.index = (self.index + delta) % len(self.rows)
        self.selected_polygon = 0
        self.show_current()

    def previous(self, _event=None) -> None:
        self._navigate(-1)

    def next(self, _event=None) -> None:
        self._navigate(1)

    def close(self, _event=None) -> None:
        if self.dirty:
            self._write_draft()
        self.window.destroy()

    def run(self) -> None:
        self.window.mainloop()


def run_annotator(root: str | Path, image_id: str | None = None) -> None:
    BruiseAnnotator(root, image_id).run()

