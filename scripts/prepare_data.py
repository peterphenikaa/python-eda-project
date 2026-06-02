"""
Chuyển dataset Tiki (raw .txt) sang CSV thống nhất cho pipeline ML.
Nguồn: https://github.com/quangvh23/tiki-dataset-sentiment-classification
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = [
    "product_id",
    "reviewer",
    "rating",
    "title",
    "helpful_votes",
    "verified",
    "time_ago",
    "text",
]

RATING_TO_LABEL = {
    20: "tieu_cuc",
    40: "tieu_cuc",
    60: "trung_lap",
    80: "tich_cuc",
    100: "tich_cuc",
}


def rating_to_sentiment(rating: int | str) -> str:
    return RATING_TO_LABEL[int(rating)]


def load_raw_folder(raw_dir: Path, variant: str = "1a3") -> pd.DataFrame:
    """Đọc toàn bộ file .txt trong thư mục variant (1a3 = text gốc, 1a6 = đã token hóa)."""
    # 1. Tạo mảng rỗng (list) chứa các object (dict), lát sẽ gom tất cả review vào đây
    rows: list[dict] = []
    
    # 2. Tạo đường dẫn gốc (ví dụ: data/raw/tiki.../1a3)
    source_dir = raw_dir / variant

    # 3. Lục lọi tất cả file .txt trong thư mục và sắp xếp theo tên (A-Z) để chạy ổn định
    for file_path in sorted(source_dir.glob("*.txt")):
        # 4. Lấy tên file bỏ đuôi .txt đi (ví dụ file TaiNghe.txt thì lấy chữ "TaiNghe") làm tên danh mục
        category = file_path.stem
        
        # 5. Mở file để đọc, dùng utf-8 để không lỗi font tiếng Việt, tự động đóng file (with)
        with file_path.open(encoding="utf-8") as handle:
            # 6. Đọc từng dòng trong file
            for line in handle:
                # Cắt khoảng trắng vô duyên ở hai đầu câu
                line = line.strip()
                # Nếu là dòng trống (không có chữ) thì bỏ qua
                if not line:
                    continue
                
                # 7. CHẶT DỮ LIỆU: Cắt câu thành mảng các phần tử tại các vị trí có dấu Tab (\t)
                parts = line.split("\t")
                # Nếu mảng bị lỗi, không đủ 8 thông tin cơ bản thì bỏ qua luôn (chống rác)
                if len(parts) < 8:
                    continue
                
                # 8. GHÉP CẶP: Ghép 8 tên cột với 8 mảnh dữ liệu, biến thành 1 Object (dict) hoàn chỉnh
                row = dict(zip(COLUMNS, parts[:8]))
                
                # Bổ sung thông tin nguồn gốc: Dòng review này lấy từ file nào (category) và thư mục nào (variant)
                row["category"] = category
                row["variant"] = variant
                
                # 9. Quăng Object này vào cái mảng khổng lồ đã tạo lúc đầu
                rows.append(row)

    # 10. HÓA PHÉP: Ép toàn bộ cái mảng khổng lồ đó thành Bảng dữ liệu (DataFrame) - Excel ảo
    df = pd.DataFrame(rows)
    
    # 11. DỌN DẸP ÉP KIỂU: 
    # Ép cột rating từ chữ ("100") thành số nguyên (100)
    df["rating"] = df["rating"].astype(int)
    
    # Ép cột helpful_votes thành số. Nếu lỗi chữ bậy bạ (coerce) thì ép thành rỗng (NaN), rồi nhét số 0 vào, ép thành số nguyên (int)
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    
    # Đổi chữ "True"/"False" thành kiểu Đúng/Sai (Boolean) của máy tính
    df["verified"] = df["verified"].map({"True": True, "False": False})
    
    # 12. TẠO CỘT MỚI: Thảy điểm rating vào hàm rating_to_sentiment, ói ra chữ "tich_cuc"/"tieu_cuc" rồi lưu vào cột mới "label"
    df["label"] = df["rating"].map(rating_to_sentiment)
    
    # 13. TRẢ HÀNG: Trả cái Bảng (Excel ảo) sạch sẽ này ra ngoài cho thằng khác dùng
    return df


def main() -> None:
    # 1. TẠO BẢNG ĐIỀU KHIỂN (CLI)
    # Khởi tạo công cụ argparse giúp file này có thể nhận lệnh từ terminal (ví dụ: python prepare_data.py --variant 1a6)
    parser = argparse.ArgumentParser(description="Prepare Tiki review dataset")
    
    # Thiết lập "Nút bấm" thứ 1: --raw-dir (Đường dẫn chứa dữ liệu thô)
    # Nếu người dùng không gõ gì, mặc định (default) tự lấy thư mục data/raw/...
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/tiki-dataset-sentiment-classification"),
        help="Thư mục repo dataset đã clone",
    )
    
    # Thiết lập "Nút bấm" thứ 2: --variant (Chọn thư mục con 1a3 hay 1a6)
    # Bắt buộc người dùng chỉ được chọn (choices) 1 trong 2 cái này. Mặc định là 1a3.
    parser.add_argument(
        "--variant",
        choices=["1a3", "1a6"],
        default="1a3",
        help="1a3: văn bản gốc | 1a6: văn bản đã tách token bằng dấu _",
    )
    
    # Thiết lập "Nút bấm" thứ 3: --output (Tên file xuất ra)
    # Mặc định sẽ lưu thành file data/tiki_reviews.csv
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tiki_reviews.csv"),
        help="File CSV đầu ra",
    )
    
    # Thu thập tất cả các "nút bấm" (tham số) mà người dùng vừa thiết lập ở trên vào biến args
    args = parser.parse_args()

    # 2. GỌI HÀM XỬ LÝ CHÍNH
    # Gọi hàm load_raw_folder (mà ta vừa học ở trên) đi cày dữ liệu, truyền vào đường dẫn và thư mục (1a3) đã cấu hình
    df = load_raw_folder(args.raw_dir, args.variant)
    
    # 3. LƯU THÀNH FILE CSV
    # Lấy thư mục cha của file đầu ra (tức là thư mục "data/") và yêu cầu tạo thư mục đó trên ổ cứng. 
    # exist_ok=True nghĩa là nếu thư mục đó có sẵn rồi thì thôi bỏ qua, không báo lỗi.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Lưu cái DataFrame df thành file CSV. 
    # index=False: nghĩa là không lưu cái cột đánh số thứ tự (0, 1, 2...) thừa thãi ở đầu dòng.
    # encoding="utf-8-sig": Mẹo cực hay giúp Excel mở file CSV lên đọc tiếng Việt ngon ơ, không bị lỗi font.
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    # 4. IN BÁO CÁO RA MÀN HÌNH TERMINAL
    # In ra dòng chữ: Đã lưu xxxx dòng vào file yyyy (dùng dấu :, để tự động phẩy ngăn cách hàng nghìn cho dễ đọc)
    print(f"Saved {len(df):,} rows -> {args.output}")
    
    print("\nLabel distribution:")
    # Lấy cột "label", đếm xem có bao nhiêu tích cực/tiêu cực/trung lập (value_counts), ép thành chữ (to_string) rồi in ra
    print(df["label"].value_counts().to_string())
    
    print("\nRating distribution:")
    # Lấy cột "rating", đếm số lượng, nhưng nhớ sắp xếp theo thứ tự điểm (sort_index: từ 20, 40, 60...) rồi mới in ra
    print(df["rating"].value_counts().sort_index().to_string())


# CÂU THẦN CHÚ KINH ĐIỂN CỦA PYTHON:
# Nếu file này được chạy trực tiếp bằng tay (kiểu: gõ lệnh 'python prepare_data.py' ở terminal) thì mới thực thi hàm main().
# Còn nếu file này bị đem đi import vào một file khác (kiểu: gõ 'import prepare_data' ở bên file khác) thì nó im re, không được tự động chạy hàm main().
if __name__ == "__main__":
    main()
