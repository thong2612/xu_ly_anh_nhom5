# MangoResearch

MangoResearch là đề tài nghiên cứu và demo phân tích màu sắc của xoài, phân loại độ chín và tình trạng khuyết tật từ ảnh. Demo chạy inference bằng các pipeline đã huấn luyện; cài đặt demo không tải dataset và không huấn luyện model.

## Phương pháp

Pipeline so sánh ảnh màu gốc với hiệu chỉnh Gray World, trích xuất đặc trưng HSV/CIELAB, tạo fruit mask bằng segmentation heuristic, và dùng K-Means trong CIELAB để khoanh vùng màu nghi vấn. Các đặc trưng được đưa vào mô hình Machine Learning cho hai tác vụ: phân loại độ chín và healthy/defective. K-Means overlay là kết quả độc lập, không phải chẩn đoán bệnh.

## Cấu trúc chính

```text
app.py                         Giao diện demo Streamlit
src/color/                     Gray World, HSV và CIELAB
src/segmentation/              Fruit-mask baseline và segmentation pipeline
src/features/                  Feature extraction, gồm K-Means
src/models/                    Huấn luyện và đánh giá Machine Learning
src/demo/                      Inference dùng chung cho demo
src/dataset/                   Tải, kiểm kê, xử lý và chia dataset
src/validation/                Công cụ đánh giá và kiểm chứng
scripts/                       Entry point nghiên cứu và training
config.yaml                    Cấu hình các pipeline
data/features/feature_schema.json
                               Thứ tự và schema feature dùng bởi model
tests/                         Kiểm thử hiện có
demo_samples/                  Hai ảnh mẫu có ghi nguồn và giấy phép
reports/phase08/               Báo cáo cuối, tài liệu tham khảo và hình
```

## Yêu cầu

- Windows x64
- Python 3.12 (64-bit)

## Cài dependencies và chạy demo

Mở PowerShell tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Trình duyệt mở giao diện tại địa chỉ Streamlit được in trong terminal. Có thể tải ảnh JPG/PNG để phân tích; hai ảnh mẫu có giấy phép được đặt trong `demo_samples/`.

### Checkpoint

Hiện chưa có link Google Drive checkpoint. Repository không chứa checkpoint và không tự tải model. Để inference hoạt động, người dùng cần tự cung cấp đúng hai file:

```text
models/phase051/ripeness_selected.joblib
models/phase051/defect_selected.joblib
```

Không cần dataset hoặc training để chạy demo khi đã có đủ hai file này. Thư mục `models/` được loại khỏi Git.

## Tải dataset và tái lập nghiên cứu (tùy chọn)

Các nguồn đang được cấu hình trong project:

- [Alphonso Mango Ripening Stage Classification Dataset — Mendeley Data](https://data.mendeley.com/datasets/tyghd6gxw2/1)
- [MangoFruitDefect — Mendeley Data](https://data.mendeley.com/datasets/jnp9tmyrt6/1)
- [Mango classification and segmentation dataset in Uganda — Mendeley Data](https://data.mendeley.com/datasets/cy3ccgkx3n/1)

Chỉ chạy pipeline dữ liệu khi chủ động muốn tải và xử lý nguồn tương ứng. Ví dụ, pipeline dưới đây tải và xử lý riêng Dataset A; các nguồn lớn khác có thể chọn riêng bằng key `mango_fruit_defect` hoặc `mango_segmentation`:

```powershell
python -m src.dataset.cli pipeline --dataset mango_ripeness
```

Sau khi tạo manifest/split cần thiết, các entry point hiện có gồm:

```powershell
python -m src.dataset.cli phase02
python -m src.dataset.cli phase1
python -m src.dataset.cli phase4-features
python -m src.dataset.cli phase5-train
python scripts/run_phase051.py --help
```

Phase 5.1 có bước rà soát near-duplicate và chuẩn bị split; hãy kiểm tra các bước review trước khi huấn luyện. Tải dataset, tạo feature và huấn luyện hoàn toàn tách khỏi lệnh chạy demo.

## Kết quả và giới hạn

Trên test split provisional của Phase 5.1, Gray World + KNN đạt Accuracy/Macro-F1 **0,9018/0,9021** cho độ chín (`n=112`); Original Color + SVM-RBF đạt **0,9630/0,9626** cho healthy/defective (`n=540`). Các split đều có `official_accuracy_eligible=false` và còn rủi ro lineage/near-duplicate, nên đây không phải ước lượng hiệu năng tổng quát. Fruit mask là heuristic; K-Means chỉ đánh dấu vùng màu nghi vấn. External evaluation cũng bị giới hạn bởi coverage và khác biệt domain.

Báo cáo khoa học cuối cùng: [MangoResearch Scientific Report — Phase 8](reports/phase08/mangoresearch_scientific_report.md). Danh mục tài liệu tham khảo và hình báo cáo nằm cùng thư mục.

## Kiểm thử

```powershell
python -m pytest -q
```

Hai ảnh mẫu trong `demo_samples/` được phân phối theo giấy phép nguồn; xem [ghi nhận nguồn và giấy phép](demo_samples/README.md).
