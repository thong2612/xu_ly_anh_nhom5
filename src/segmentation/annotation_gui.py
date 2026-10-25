from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from src.color.preprocess import decode_image
from src.utils.config import project_path


def heuristic_mask_to_polygon(mask: np.ndarray, max_vertices: int = 120) -> list[tuple[int, int]]:
    """Convert the largest binary heuristic component to a coarse editable contour."""
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
        raise ValueError("Heuristic suggestion must be a 2D binary uint8 0/255 mask")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    while len(polygon) > max_vertices and epsilon < 0.05 * cv2.arcLength(contour, True):
        epsilon *= 1.5
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [(int(x), int(y)) for x, y in polygon]


class FruitMaskAnnotator:
    """Compact polygon editor with resumable drafts and explicit candidate confirmation."""

    COMPLETE_STATUS = "annotated"  # only set by the separate semantic approval command

    def __init__(self, config: dict, limit: int | None = None) -> None:
        self.config = config
        self.root_path = Path(config["_project_root"])
        self.candidates_path = project_path(config, "data/metadata/fruit_gt_candidates.csv")
        self.draft_dir = project_path(config, "data/annotations/fruit_drafts")
        self.output_dir = project_path(config, "data/annotations/fruit_masks")
        self.session_path = self.draft_dir / "session.json"
        self.rows = self._read_candidates()
        if limit is not None:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError("No unapproved whole-fruit candidates are available")

        self.index = self._resume_index()
        self.points: list[tuple[int, int]] = []
        self.suggestion_points: list[tuple[int, int]] = []
        self.undo_stack: list[list[tuple[int, int]]] = []
        self.active_vertex: int | None = None
        self.scale = 1.0
        self.dirty = False
        self.add_vertex_mode = False
        self.tk_image = None
        self.root = tk.Tk()
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.max_w, self.max_h = min(1160, screen_w - 80), min(670, screen_h - 240)
        self.root.title("MangoResearch | Chú giải mask toàn quả")
        self.root.geometry(f"{min(screen_w - 30, 1240)}x{min(screen_h - 50, 880)}")
        self.root.minsize(820, 560)

        header = tk.Frame(self.root, padx=12, pady=7)
        header.pack(fill="x")
        self.status = tk.StringVar()
        tk.Label(header, textvariable=self.status, anchor="w", font=("Segoe UI", 10, "bold")).pack(fill="x")
        tk.Label(header, text="Xanh: mask đang sửa  |  Đỏ nét đứt: heuristic gợi ý  |  Mask chỉ là ứng viên; chưa được approve",
                 anchor="w", fg="#4a4a4a", font=("Segoe UI", 9)).pack(fill="x", pady=(4, 0))

        self.canvas = tk.Canvas(self.root, background="#20252b", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_remove_vertex)

        controls = tk.Frame(self.root, padx=8, pady=8)
        controls.pack(fill="x")
        self._button(controls, "← Trước", self.previous).pack(side="left", padx=3)
        self._button(controls, "Tiếp →", self.next).pack(side="left", padx=3)
        self._button(controls, "Nạp lại heuristic", self.use_heuristic_suggestion).pack(side="left", padx=3)
        self.add_vertex_button = self._button(controls, "Thêm đỉnh: TẮT", self.toggle_add_vertex_mode)
        self.add_vertex_button.pack(side="left", padx=3)
        self._button(controls, "Xóa polygon", self.reset).pack(side="left", padx=3)
        self._button(controls, "Undo", self.undo).pack(side="left", padx=3)
        self._button(controls, "Lưu nháp", self.save_draft).pack(side="right", padx=3)
        self._button(controls, "Cần sửa", self.mark_needs_edit).pack(side="right", padx=3)
        self._button(controls, "Xác nhận mask & lưu ứng viên", self.save_candidate,
                     background="#d8f3e6").pack(side="right", padx=3)
        footer = tk.Label(self.root,
                          text="Kéo chấm xanh để chỉnh; bấm sát cạnh để chèn đỉnh. Bật ‘Thêm đỉnh’ để vẽ tiếp; chuột phải lên đỉnh để xóa. Enter: lưu ứng viên · A: thêm đỉnh · H: heuristic · U: undo · ←/→: chuyển · Esc: lưu và thoát",
                          anchor="w", padx=12, pady=4, fg="#555", font=("Segoe UI", 9))
        footer.pack(fill="x")
        self.root.bind("<Return>", self.save_candidate)
        self.root.bind("a", self.toggle_add_vertex_mode)
        self.root.bind("h", self.use_heuristic_suggestion)
        self.root.bind("u", self.undo)
        self.root.bind("<Left>", self.previous)
        self.root.bind("<Right>", self.next)
        self.root.bind("<Escape>", self.close)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.show_current()

    @staticmethod
    def _button(parent: tk.Misc, text: str, command, background: str = "#eeeeee") -> tk.Button:
        return tk.Button(parent, text=text, command=command, padx=8, pady=5,
                         font=("Segoe UI", 9), background=background, activebackground="#cdeee0")

    def _read_candidates(self) -> list[dict[str, str]]:
        with self.candidates_path.open("r", encoding="utf-8-sig", newline="") as stream:
            all_rows = list(csv.DictReader(stream))
        return [row for row in all_rows if row.get("annotation_status") != self.COMPLETE_STATUS]

    def _resume_index(self) -> int:
        if self.session_path.is_file():
            try:
                session = json.loads(self.session_path.read_text(encoding="utf-8"))
                image_id = session.get("current_image_id")
                for index, row in enumerate(self.rows):
                    if row.get("image_id") == image_id:
                        return index
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        # Resume unfinished edits before reopening already-saved candidates.
        for index, row in enumerate(self.rows):
            if row.get("annotation_status") in {"in_progress", "in_progress_needs_edit", "pending_manual_gt"}:
                return index
        return 0

    def _save_session(self, image_id: str | None = None) -> None:
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        data = {"current_image_id": image_id if image_id is not None else self.rows[self.index]["image_id"],
                "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
        temp = self.session_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.session_path)

    def _write_draft(self, status: str = "in_progress") -> None:
        row = self.rows[self.index]
        self.draft_dir.mkdir(parents=True, exist_ok=True)
        target = self.draft_dir / f"{row['image_id']}.json"
        payload = {"image_id": row["image_id"], "source_image_path": row["source_image_path"],
                   "width": int(self.frame.rgb.shape[1]), "height": int(self.frame.rgb.shape[0]),
                   "points": [[int(x), int(y)] for x, y in self.points],
                   "draft_status": status, "approved": False}
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)
        self._update_candidate_status(status, "Draft polygon autosaved; not approved.")
        self.dirty = False

    def _update_candidate_status(self, status: str, note: str) -> None:
        with self.candidates_path.open("r", encoding="utf-8-sig", newline="") as stream:
            candidates = list(csv.DictReader(stream))
        fields = list(candidates[0]) if candidates else []
        for candidate in candidates:
            if candidate.get("image_id") == self.rows[self.index]["image_id"]:
                candidate["annotation_status"] = status
                candidate["notes"] = note
        if fields:
            temp = self.candidates_path.with_suffix(".csv.tmp")
            with temp.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(candidates)
            temp.replace(self.candidates_path)
        self.rows[self.index]["annotation_status"] = status

    def _load_heuristic(self, row: dict[str, str]) -> tuple[list[tuple[int, int]], np.ndarray | None]:
        suggestion_path = project_path(self.config,
                                       f"data/processed/fruit_masks/{row['dataset']}/E1/{row['image_id']}.png")
        suggestion = cv2.imread(str(suggestion_path), cv2.IMREAD_GRAYSCALE) if suggestion_path.is_file() else None
        if suggestion is None or suggestion.shape != self.frame.rgb.shape[:2] or not np.isin(suggestion, (0, 255)).all():
            return [], None
        return heuristic_mask_to_polygon(suggestion), suggestion

    def _load_existing_png(self, row: dict[str, str]) -> list[tuple[int, int]]:
        possible = [self.output_dir / f"{row['image_id']}.png"]
        candidate_path = row.get("candidate_ground_truth_path", "")
        if candidate_path:
            possible.insert(0, project_path(self.config, candidate_path))
        for path in possible:
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
            if mask is not None and mask.shape == self.frame.rgb.shape[:2] and np.isin(mask, (0, 255)).all():
                return heuristic_mask_to_polygon(mask)
        return []

    def _load_draft(self, row: dict[str, str]) -> list[tuple[int, int]] | None:
        path = self.draft_dir / f"{row['image_id']}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("image_id") != row["image_id"] or data.get("width") != self.frame.rgb.shape[1]
                    or data.get("height") != self.frame.rgb.shape[0]):
                return None
            width, height = self.frame.rgb.shape[1], self.frame.rgb.shape[0]
            return [(min(max(int(x), 0), width - 1), min(max(int(y), 0), height - 1))
                    for x, y in data.get("points", [])]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _progress_text(self) -> str:
        statuses = [row.get("annotation_status", "pending_manual_gt") for row in self._read_candidates_all()]
        saved = sum(value in {"annotation_saved_pending_import", "imported_pending_semantic_review", "annotated"} for value in statuses)
        draft = sum(value in {"in_progress", "in_progress_needs_edit"} for value in statuses)
        return f"Ứng viên đã lưu: {saved}/{len(statuses)}  |  Đang nháp: {draft}"

    def _read_candidates_all(self) -> list[dict[str, str]]:
        with self.candidates_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def show_current(self) -> None:
        row = self.rows[self.index]
        self.frame = decode_image(project_path(self.config, row["source_image_path"]))
        rgb = self.frame.rgb
        image = Image.fromarray(rgb)
        self.scale = min(self.max_w / image.width, self.max_h / image.height, 1.0)
        shown = image.resize((max(1, round(image.width * self.scale)), max(1, round(image.height * self.scale))),
                             Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(shown)
        self.canvas.config(width=shown.width, height=shown.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.undo_stack.clear()
        self.active_vertex = None
        self.suggestion_points, self.suggestion_mask = self._load_heuristic(row)
        saved_points = self._load_draft(row)
        if saved_points is None:
            saved_points = self._load_existing_png(row)
        self.points = saved_points if saved_points is not None else list(self.suggestion_points)
        self.add_vertex_mode = len(self.points) < 3
        self._update_add_vertex_button()
        self.dirty = False
        self.redraw_polygon()
        state = row.get("annotation_status", "pending_manual_gt")
        source = "heuristic draft" if self.points == self.suggestion_points and self.suggestion_points else "saved draft/mask"
        if not self.points:
            source = "no valid heuristic; draw polygon"
        self.status.set(f"{self.index + 1}/{len(self.rows)}  |  {self._progress_text()}  |  "
                        f"{row['dataset']} · {row['task']} · {row['label']}  |  {row['image_id']}  |  "
                        f"{image.width}×{image.height}  |  {state}  |  {source}")
        self._save_session(row["image_id"])

    def _image_point(self, x: int, y: int) -> tuple[int, int]:
        h, w = self.frame.rgb.shape[:2]
        return (min(max(int(round(x / self.scale)), 0), w - 1),
                min(max(int(round(y / self.scale)), 0), h - 1))

    def _snapshot(self) -> None:
        self.undo_stack.append(list(self.points))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    @staticmethod
    def _point_segment_distance(point: tuple[float, float], a: tuple[float, float],
                                b: tuple[float, float]) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        denominator = dx * dx + dy * dy
        t = 0.0 if denominator == 0 else min(max(((px - ax) * dx + (py - ay) * dy) / denominator, 0.0), 1.0)
        return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))

    def on_press(self, event: tk.Event) -> None:
        display_points = [(x * self.scale, y * self.scale) for x, y in self.points]
        if display_points:
            distances = [np.hypot(event.x - x, event.y - y) for x, y in display_points]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 11:
                self._snapshot()
                self.active_vertex = nearest
                return
            if len(display_points) >= 2:
                edges = [(self._point_segment_distance((event.x, event.y), display_points[i],
                                                       display_points[(i + 1) % len(display_points)]), i)
                         for i in range(len(display_points))]
                distance, edge_index = min(edges)
                if distance <= 10 or self.add_vertex_mode:
                    self._snapshot()
                    self.active_vertex = edge_index + 1
                    self.points.insert(self.active_vertex, self._image_point(event.x, event.y))
                    self.dirty = True
                    self.redraw_polygon()
                    return
            if self.add_vertex_mode:
                self._snapshot()
                self.points.append(self._image_point(event.x, event.y))
                self.dirty = True
                self.redraw_polygon()
                return
            self.status.set("Bấm sát đường viền để thêm đỉnh, hoặc kéo một chấm xanh để chỉnh.")
            return
        self._snapshot()
        self.points.append(self._image_point(event.x, event.y))
        self.dirty = True
        self.redraw_polygon()

    def on_drag(self, event: tk.Event) -> None:
        if self.active_vertex is None or self.active_vertex >= len(self.points):
            return
        self.points[self.active_vertex] = self._image_point(event.x, event.y)
        self.dirty = True
        self.redraw_polygon()

    def on_release(self, _event: tk.Event) -> None:
        self.active_vertex = None
        if self.dirty:
            self._write_draft()
            self.show_current_status()

    def on_remove_vertex(self, event: tk.Event) -> None:
        if not self.points:
            return
        shown = [(x * self.scale, y * self.scale) for x, y in self.points]
        distances = [np.hypot(event.x - x, event.y - y) for x, y in shown]
        nearest = int(np.argmin(distances))
        if distances[nearest] > 14:
            return
        self._snapshot()
        self.points.pop(nearest)
        self.dirty = True
        self.redraw_polygon()
        self._write_draft("in_progress_needs_edit" if len(self.points) < 3 else "in_progress")
        self.show_current_status()

    def _update_add_vertex_button(self) -> None:
        button = getattr(self, "add_vertex_button", None)
        if button is not None:
            button.config(text=f"Thêm đỉnh: {'BẬT' if self.add_vertex_mode else 'TẮT'}")

    def toggle_add_vertex_mode(self, _event: tk.Event | None = None) -> None:
        self.add_vertex_mode = not self.add_vertex_mode
        self._update_add_vertex_button()
        if hasattr(self, "status"):
            self.show_current_status()

    def show_current_status(self) -> None:
        row = self.rows[self.index]
        self.status.set(f"{self.index + 1}/{len(self.rows)}  |  {self._progress_text()}  |  "
                        f"{row['image_id']}  |  Nháp đã lưu tự động; chưa được approve")

    def redraw_polygon(self) -> None:
        self.canvas.delete("suggestion")
        self.canvas.delete("annotation")
        if len(self.suggestion_points) > 1:
            suggestion = [(round(x * self.scale), round(y * self.scale)) for x, y in self.suggestion_points]
            self.canvas.create_line(*[v for p in suggestion for v in p], fill="#ff4a3d", width=2,
                                    dash=(6, 4), tags="suggestion")
        shown = [(round(x * self.scale), round(y * self.scale)) for x, y in self.points]
        if len(shown) >= 3:
            self.canvas.create_polygon(*[v for p in shown for v in p], fill="#25c982", stipple="gray50",
                                       outline="#19ed91", width=2, tags="annotation")
        elif len(shown) >= 2:
            self.canvas.create_line(*[v for p in shown for v in p], fill="#19ed91", width=2, tags="annotation")
        for x, y in shown:
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="#19ed91", outline="#153d2d",
                                    width=1, tags="annotation")

    def undo(self, _event: tk.Event | None = None) -> None:
        if self.undo_stack:
            self.points = self.undo_stack.pop()
            self.dirty = True
            self.redraw_polygon()
            self._write_draft()
            self.show_current_status()

    def reset(self, _event: tk.Event | None = None) -> None:
        if not messagebox.askyesno("Xóa polygon", "Xóa mask đang sửa? Bản heuristic vẫn có thể nạp lại."):
            return
        self._snapshot()
        self.points = []
        self.dirty = True
        self.redraw_polygon()
        self._write_draft("in_progress_needs_edit")
        self.show_current_status()

    def use_heuristic_suggestion(self, _event: tk.Event | None = None) -> None:
        if len(self.suggestion_points) < 3:
            messagebox.showinfo("Không có gợi ý", "Ảnh này không có E1 heuristic mask hợp lệ.")
            return
        if not messagebox.askyesno("Nạp heuristic draft?",
                                   "Nạp lại đường viền thuật toán làm bản nháp. Bạn cần kiểm tra và chỉnh sửa theo ảnh RGB gốc."):
            return
        self._snapshot()
        self.points = list(self.suggestion_points)
        self.dirty = True
        self.redraw_polygon()
        self._write_draft()
        self.show_current_status()

    def save_draft(self, _event: tk.Event | None = None) -> None:
        if len(self.points) < 3:
            messagebox.showwarning("Polygon chưa đủ", "Cần ít nhất 3 đỉnh để lưu nháp.")
            return
        self._write_draft()
        self.show_current_status()
        messagebox.showinfo("Đã lưu nháp", "Tiến độ đã lưu. Đây chưa phải mask được duyệt.")

    def mark_needs_edit(self, _event: tk.Event | None = None) -> None:
        if self.points:
            self._write_draft("in_progress_needs_edit")
        else:
            self._update_candidate_status("in_progress_needs_edit", "Human review marked this image as needing annotation; not approved.")
        self._advance(1)

    def save_candidate(self, _event: tk.Event | None = None) -> None:
        if len(self.points) < 3:
            messagebox.showwarning("Polygon chưa đủ", "Cần ít nhất 3 đỉnh để lưu ứng viên.")
            return
        if not messagebox.askyesno("Xác nhận mask toàn quả",
                                   "Bạn đã kiểm tra ảnh RGB gốc và xác nhận polygon bao đúng toàn bộ phần quả xoài nhìn thấy?\n\n"
                                   "Thao tác này chỉ lưu ứng viên. Mask vẫn chưa được approve; cần bước import và semantic approval riêng."):
            return
        mask = Image.new("L", (self.frame.rgb.shape[1], self.frame.rgb.shape[0]), 0)
        ImageDraw.Draw(mask).polygon(self.points, fill=255)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        row = self.rows[self.index]
        mask.save(self.output_dir / f"{row['image_id']}.png", format="PNG")
        self._write_draft("annotation_saved_pending_import")
        self._update_candidate_status("annotation_saved_pending_import",
                                      "User confirmed whole-fruit candidate in annotator; import and semantic approval are still required.")
        self._advance(1)

    def _advance(self, direction: int) -> None:
        if self.dirty:
            self._write_draft()
        next_index = self.index + direction
        if next_index < 0:
            next_index = 0
        if next_index >= len(self.rows):
            self._save_session()
            messagebox.showinfo("Đã tới cuối danh sách", "Tiến độ đã được lưu. Các ứng viên vẫn chưa được approve.")
            return
        self.index = next_index
        self.show_current()

    def next(self, _event: tk.Event | None = None) -> None:
        self._advance(1)

    def previous(self, _event: tk.Event | None = None) -> None:
        self._advance(-1)

    def close(self, _event: tk.Event | None = None) -> None:
        if self.dirty:
            self._write_draft()
        self._save_session()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def annotate_fruit_masks(config: dict, limit: int | None = None) -> None:
    FruitMaskAnnotator(config, limit).run()

