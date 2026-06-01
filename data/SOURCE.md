# Nguồn dữ liệu — Tiki Product Reviews

## Thông tin dataset

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | [quangvh23/tiki-dataset-sentiment-classification](https://github.com/quangvh23/tiki-dataset-sentiment-classification) |
| **Loại** | Đánh giá sản phẩm thương mại điện tử Tiki (tiếng Việt) |
| **Số mẫu** | 13.652 review |
| **Số nhãn** | 3 (sentiment: tiêu cực / trung lập / tích cực) |
| **Danh mục sản phẩm** | 10 category (Bút viết, Tai nghe, Trang điểm, ...) |

## Cấu trúc file gốc

Repo gồm 2 biến thể văn bản:

- **`1a3/`** — văn bản review **gốc** (dùng cho tiền xử lý NLP trong bài tập)
- **`1a6/`** — văn bản đã **tách token** bằng dấu `_` (dùng tham khảo)

Mỗi file `.txt` tương ứng một danh mục sản phẩm, mỗi dòng là một review, phân tách bằng tab (`\t`):

| Cột | Mô tả |
|-----|--------|
| `product_id` | Mã/slug sản phẩm |
| `reviewer` | Tên người đánh giá |
| `rating` | Điểm sao Tiki (20, 40, 60, 80, 100) |
| `title` | Tiêu đề/tóm tắt review |
| `helpful_votes` | Số vote hữu ích |
| `verified` | Đã mua hàng (`True`/`False`) |
| `time_ago` | Thời gian đăng (vd: "3 tháng trước") |
| `text` | Nội dung review |

## Ánh xạ nhãn phân loại (3 class)

Điểm sao Tiki được quy về 3 nhãn cảm xúc:

| Rating | Nhãn | Ý nghĩa |
|--------|------|---------|
| 20, 40 | `tieu_cuc` | Tiêu cực |
| 60 | `trung_lap` | Trung lập |
| 80, 100 | `tich_cuc` | Tích cực |

## File đã xử lý

- `tiki_reviews.csv` — toàn bộ review gộp từ `1a3/`, có thêm cột `category`, `label`

## Tái tạo dữ liệu

```bash
git clone https://github.com/quangvh23/tiki-dataset-sentiment-classification.git data/raw/tiki-dataset-sentiment-classification
python scripts/prepare_data.py
```

## Ghi chú sử dụng học thuật

Dataset thu thập từ đánh giá công khai trên Tiki, dùng cho mục đích học tập. Nhóm ghi rõ nguồn trong báo cáo và notebook.
