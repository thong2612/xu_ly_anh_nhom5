# MangoResearch

## SETUP & RUN DEMO

Cài Python 3.12, clone repository và cài dependencies:

```powershell
git clone https://github.com/thong2612/xu_ly_anh_nhom5.git
Set-Location xu_ly_anh_nhom5
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Checkpoint không nằm trong GitHub và hiện chưa có link tải trong repository. Đặt hai file được cấp riêng tại `models/classifiers/ripeness_selected.joblib` và `models/classifiers/defect_selected.joblib`, rồi chạy demo:

```powershell
python -m streamlit run app.py
```

## DOWNLOAD DATASET & TRAIN

Từ thư mục repository, tải và chuẩn bị hai nguồn cần cho training, tạo feature, rồi audit các cặp ảnh trùng gần. Xem contact sheets và lưu vào `reports/model_selection/pair_review_decisions.csv` với các cột `task,image_id_a,image_id_b,review_decision,review_notes`; quyết định dùng `confirmed_same_source` hoặc `rejected_different_fruit` trước khi tạo split và train:

```powershell
python -m src.dataset.cli pipeline --dataset mango_ripeness
python -m src.dataset.cli pipeline --dataset mango_segmentation
python -m src.dataset.cli curate
python -m src.dataset.cli extract-features
python scripts/train.py audit
# Review contact sheets; save decisions to reports/model_selection/pair_review_decisions.csv
python scripts/train.py review
python scripts/train.py split
python scripts/train.py train
```
