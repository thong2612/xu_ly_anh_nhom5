from __future__ import annotations

import logging
import hashlib
from pathlib import Path

import streamlit as st

from src.demo.phase6 import InputImageError, infer_image, load_demo_assets


ROOT = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="MangoResearch · Demo", page_icon="🥭", layout="wide")


@st.cache_resource(show_spinner="Đang nạp model Phase 5.1…")
def _assets():
    return load_demo_assets(ROOT)


def _label_vi(task: str, label: str) -> str:
    translations = {
        "unripe": "Chưa chín", "partially_ripe": "Chín một phần", "ripe": "Chín",
        "healthy": "Healthy · Không có nhãn khuyết tật",
        "defective": "Defective · Có nhãn khuyết tật",
    }
    return translations.get(label, label)


def _prediction_card(title: str, result: dict) -> None:
    st.subheader(title)
    if not result["available"]:
        st.warning(f"Chưa khả dụng: {result['reason']}")
        return
    st.markdown(f"### {_label_vi(title, result['prediction'])}")
    probabilities = result.get("probabilities")
    if probabilities:
        st.caption("Xác suất thô từ classifier, chưa được hiệu chuẩn thành độ tin cậy.")
        for label, value in probabilities.items():
            st.write(f"{_label_vi(title, label)} · {value:.1%}")
    st.caption(f"Đặc trưng thiếu: {result['missing_features']}/{result['feature_count']} · pipeline đã fit dùng imputer đã lưu.")


def _show_image(title: str, image, caption: str | None = None, clamp: bool = False) -> None:
    st.image(image, caption=caption, width="stretch", clamp=clamp)
    st.caption(title)


st.title("🥭 MangoResearch")
st.caption("Demo nghiên cứu về độ chín và vùng màu nghi vấn trên quả xoài")
st.info("Đây là demo nghiên cứu. Chưa có bộ kiểm tra chắc chắn ảnh có chứa xoài; ảnh không có xoài có thể cho kết quả không đáng tin cậy.")

uploaded = st.file_uploader("Chọn ảnh JPG hoặc PNG", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    if st.button("Phân tích ảnh", type="primary"):
        try:
            with st.spinner("Đang phân tích ảnh…"):
                assets = _assets()
                result = infer_image(uploaded.getvalue(), uploaded.name, assets)
            st.session_state["phase06_result"] = result
            st.session_state["phase06_filename"] = uploaded.name
        except InputImageError as exc:
            st.error(str(exc))
        except Exception as exc:
            logging.exception("Phase 6 inference failed")
            st.error(f"Không thể phân tích ảnh: {type(exc).__name__}. Kiểm tra model và cấu hình dự án.")

result = st.session_state.get("phase06_result")
if uploaded is None:
    result = None
elif result is not None:
    current_id = f"demo_{hashlib.sha256(uploaded.getvalue()).hexdigest()[:20]}"
    if result.get("image_id") != current_id:
        result = None
if result:
    st.divider()
    left, right = st.columns(2)
    with left:
        _prediction_card("Độ chín", result["ripeness"])
    with right:
        _prediction_card("Tình trạng khuyết tật", result["defect"])
    st.warning(result["warning"])
    st.caption(result["scientific_caveat"])

    st.subheader("Các bước xử lý")
    panels = [
        ("Ảnh gốc (thu nhỏ để hiển thị)", result["original_rgb"], "Ảnh tải lên đã được kiểm tra"),
        ("Gray World", result["gray_world_rgb"],
         f"applied={result['gray_world']['applied']} · gains={tuple(round(x, 3) for x in result['gray_world']['gains'])}"),
        ("Fruit mask heuristic · Original", result["fruit_mask"], result["fruit_mask_status"]),
        ("Ảnh sau khi loại nền", result["background_removed_rgb"], "Chỉ hiển thị khi mask hợp lệ"),
        ("K-Means cluster map · Lab, K=3", result["cluster_map_rgb"], result["kmeans"]["status"]),
        ("Mask vùng màu nghi vấn", result["suspicious_mask"], "Đây không phải ground truth"),
        ("Overlay vùng nghi vấn", result["overlay_rgb"], "Màu đỏ biểu thị pixel được K-Means chọn"),
    ]
    cols = st.columns(3)
    for index, (title, image, caption) in enumerate(panels):
        with cols[index % 3]:
            if image is not None:
                _show_image(title, image, caption, clamp=True)
            else:
                st.info(f"{title}: chưa có kết quả ({caption}).")

    st.subheader("Thông số kỹ thuật")
    ratio = result["suspicious_area_percent"]
    ratio_text = f"{ratio:.2f}%" if ratio is not None else "Không khả dụng"
    st.write(f"- Tỷ lệ vùng nghi vấn / diện tích quả nhìn thấy: **{ratio_text}**")
    st.write(f"- Fruit mask Original: **{result['fruit_mask_status']}**; nguồn: heuristic Phase 2.")
    st.write(f"- Fruit mask Gray World: **{result['gray_world_mask_status']}**.")
    st.write(f"- Gray World clipping: **{result['gray_world']['clipping_fraction']:.2%}**.")
    st.write(f"- Trích xuất màu Original: **{result['feature_status']['original_color']}**; Gray World: **{result['feature_status']['gray_world_color']}**.")
    st.write(f"- K-Means: **Original + CIELAB + K=3**; trạng thái **{result['kmeans']['status']}**; cụm nghi vấn: {result['kmeans']['selected_clusters']}.")
    st.caption("Dự đoán ML và vùng nghi vấn K-Means là hai đầu ra độc lập, có thể không đồng ý. Model defect đã chọn dùng Original Color; K-Means Lab K=3 chỉ tạo bản đồ vùng màu demo.")

