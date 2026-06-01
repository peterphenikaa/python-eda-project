# Bài Tập Lớn Cuối Kỳ — Phân tích Văn bản & Phân loại Tự động (ML)

**Thời gian thực hiện (dự kiến):** 5 ngày  
**Nhóm:** Bình + Hải (cả hai dev chính, chia việc đều mỗi ngày)  
**Deadline nộp bài (theo đề):** Cuối tuần thứ 3 kể từ ngày phát đề, trước 23:59  

---

## 1. Tổng quan dự án

Xây dựng **pipeline phân loại văn bản hoàn chỉnh**: từ thu thập/làm sạch dữ liệu → EDA → tiền xử lý → trích xuất đặc trưng → huấn luyện ≥4 mô hình → đánh giá → tối ưu hóa → báo cáo.

### Deliverables bắt buộc (cấu trúc nộp `.zip`)

```
Ten_nhom/
├── notebook.ipynb      # Jupyter Notebook đầy đủ, có output
├── report.pdf          # Báo cáo (PDF/DOCX)
└── data/               # Dataset hoặc script thu thập (nếu >50MB → link Drive)
```

### Yêu cầu dữ liệu

| Tiêu chí | Yêu cầu | Trạng thái |
|----------|---------|------------|
| Ngôn ngữ | Tiếng Anh **hoặc** Tiếng Việt | ✅ Tiếng Việt |
| Số mẫu | ≥ 5.000 | ✅ 13.652 mẫu |
| Số nhãn | ≥ 3 nhãn phân loại | ✅ 3 nhãn sentiment |
| Nguồn đã chọn | [Tiki Reviews (GitHub)](https://github.com/quangvh23/tiki-dataset-sentiment-classification) | ✅ Đã tải |

**File dữ liệu:** `data/tiki_reviews.csv` — chi tiết nguồn xem `data/SOURCE.md`

**Nhãn:** `tieu_cuc` (rating 20/40) · `trung_lap` (60) · `tich_cuc` (80/100)

### Tech stack gợi ý

- **Xử lý:** pandas, numpy  
- **NLP:** NLTK hoặc spaCy (tokenize, stopwords, lemmatization/stemming)  
- **ML:** scikit-learn  
- **Trực quan hóa:** matplotlib, seaborn, wordcloud  
- **Notebook:** Jupyter  

---

## 2. Solution phải làm gì? (Checklist theo đề)

### PHẦN A — Tiền xử lý & Phân tích văn bản (20đ)

| ID | Việc cần làm | Output |
|----|--------------|--------|
| A1 | EDA ban đầu: load data, 10 dòng đầu, mô tả cấu trúc DataFrame, phân phối nhãn, nhận xét imbalanced | Code + markdown trả lời |
| A2 | Thống kê độ dài văn bản (số từ, số ký tự) theo nhãn; so sánh phân phối; phân tích tương quan độ dài ↔ nhãn | Bảng + nhận xét |
| A3 | Hàm `preprocess_text(text)`: lowercase, bỏ HTML/URL/ký tự đặc biệt, tokenize, stopwords, stemming **hoặc** lemmatization | Hàm + giải thích từng bước |
| A3-Q | Trả lời: (a) tại sao bỏ stopwords + 5 ví dụ; (b) so sánh Stemming vs Lemmatization | Markdown trong notebook |
| A4 | So sánh **CountVectorizer** (max_features=10000) vs **TfidfVectorizer** (max_features=10000, ngram_range=(1,2)) | Code + giải thích TF-IDF |
| A6 | Phân tích lỗi code sinh viên X (fit vectorizer trên test); giải thích data leakage | Markdown |

### PHẦN B — Trực quan hóa (20đ)

| ID | Việc cần làm | Output |
|----|--------------|--------|
| B1 | Bar chart + pie chart phân phối nhãn (tiêu đề, nhãn trục, màu, chú thích) | 2 biểu đồ + nhận xét |
| B2 | Box plot độ dài văn bản theo nhãn | Biểu đồ + nhận xét |
| B4 | Word cloud **riêng từng nhãn** (≥3 nhãn), nền tương phản | ≥3 figure + nhận xét từ vựng |

### PHẦN C — Xây dựng mô hình (25đ)

| ID | Việc cần làm | Output |
|----|--------------|--------|
| C1 | Split train/val/test **70/10/20**, `stratify=y`, `random_state` cố định; in phân phối từng tập | Code + giải thích stratify & val vs test |
| C2 | Huấn luyện 4 mô hình với TF-IDF features + grid hyperparameter gợi ý | 4 mô hình đã fit |
| C2-Q | Trả lời ngắn về C (LR), margin/support vectors (SVM), overfitting (RF) | Markdown |
| C3 | Stratified K-Fold CV (k=5) cho tất cả mô hình; mean ± std accuracy | Bảng CV + 3 câu hỏi phân tích |

**4 mô hình bắt buộc:**

| Mô hình | Hyperparameter gợi ý |
|---------|----------------------|
| MultinomialNB | alpha = {0.1, 0.5, 1.0} |
| LogisticRegression | C = {0.1, 1, 10}, solver='lbfgs', max_iter=1000 |
| LinearSVC | C = {0.1, 1, 10} |
| RandomForest | n_estimators = {100, 200}, max_depth = {None, 10, 20} |

### PHẦN D — Đánh giá & Phân tích (20đ)

| ID | Việc cần làm | Output |
|----|--------------|--------|
| D1 | Bảng test: Accuracy, Precision, Recall, F1 (macro avg) cho 4 mô hình | Bảng kết quả thực nghiệm |
| D2 | `classification_report` mô hình tốt nhất; phân tích Precision/Recall theo nhãn; tình huống lọc review tiêu cực | Markdown có lập luận |
| D3 | 10 mẫu bị phân loại sai (random); phân tích đặc điểm chung + đề xuất xử lý | Bảng + phân tích |
| D4 | Phản biện: "Accuracy 85% đủ để production?" | 300–500 từ lập luận |

### PHẦN E — Tối ưu hóa (15đ)

| ID | Việc cần làm | Output |
|----|--------------|--------|
| E1 | GridSearchCV **hoặc** RandomizedSearchCV cho ≥2 mô hình tốt nhất; ≥3 tham số × ≥3 giá trị; so sánh trước/sau | Bảng tuning |
| E2 | Trả lời: Grid vs Random, tại sao CV trong tuning, diminishing returns | Markdown |
| E3 | Triển khai ≥1 kỹ thuật nâng cao (SMOTE/class_weight, Ensemble, feature engineering, TruncatedSVD) | So sánh trước/sau |
| E4 | Phản tư 300–400 từ: hướng cải tiến nếu có thêm 2 tuần + bài học rút ra | Trong report + notebook |

---

## 3. Phân công nhóm (Bình & Hải)

**Nguyên tắc:** Cả hai là **dev chính** — mỗi ngày chia task **~50/50**, không phân vai “chỉ làm data” / “chỉ làm model”.

| Nguyên tắc | Chi tiết |
|------------|----------|
| Pair work | Mỗi ngày sync 15 phút; review PR/cell notebook của nhau trước khi merge |
| Hiểu hết pipeline | Sau mỗi ngày, cả hai đọc lại phần đối phương làm — sẵn sàng vấn đáp mọi phần |
| Notebook chung | Một file `notebook.ipynb`; chia section rõ (comment `# [Bình]` / `# [Hải]`) hoặc branch ngắn merge trong ngày |
| Báo cáo | `report.pdf` viết chung; mỗi người draft phần mình làm trong ngày, gộp ngày 5 |

> **Quy tắc:** Không merge code mình không hiểu. Ngày 5 cả hai **Restart & Run All** và mock vấn đáp chéo.

---

## 4. Lộ trình 5 ngày

| Ngày | Phần đề | Trạng thái |
|------|---------|------------|
| 1 | Setup + A1 (EDA sơ bộ) | ✅ Đã xong |
| 2 | A2, A3, A4, A6 — hoàn thành Phần A | 🔲 |
| 3 | B1, B2, B4 + C1 — Trực quan hóa & chia dữ liệu | 🔲 |
| 4 | C2, C3 — 4 mô hình + Cross-Validation | 🔲 |
| 5 | D, E + báo cáo + đóng gói nộp bài | 🔲 |

---

### 📅 Ngày 1 — Thiết lập & Thu thập dữ liệu ✅

**Mục tiêu:** Có dataset sạch, repo và notebook khung sẵn sàng.

| Dev | Tasks | Trạng thái |
|-----|-------|------------|
| **Bình** | Tải dataset Tiki; `data/`, `SOURCE.md`, `prepare_data.py` → `tiki_reviews.csv`; EDA sơ bộ (head, info, value_counts) | ✅ |
| **Hải** | `requirements.txt`, `notebook.ipynb` skeleton, `random_state=42` | ✅ (hoặc gộp chung ngày 1) |
| **Chung** | Thống nhất cột `text`, `label`; NLTK cho tiếng Việt | ✅ |

**Done:** 13.652 mẫu, 3 nhãn, `data/tiki_reviews.csv`, notebook có A1.

---

### 📅 Ngày 2 — Hoàn thành Phần A (Tiền xử lý & Phân tích)

**Mục tiêu:** Xong A2 → A6; có cột `text_clean`; sẵn sàng vectorize ngày 3.

| Dev | Tasks (~50%) |
|-----|----------------|
| **Bình** | **A2:** `word_count`, `char_count` theo nhãn; so sánh phân phối; nhận xét tương quan độ dài ↔ nhãn |
| **Bình** | **A3:** `preprocess_text()` (lowercase, HTML, URL, special chars, tokenize, stopwords, lemma/stem) + giải thích từng bước |
| **Bình** | Trả lời **A3(a)(b)** trong markdown |
| **Hải** | **A4:** CountVectorizer & TfidfVectorizer (`max_features=10000`, `ngram_range=(1,2)`); so sánh BoW vs TF-IDF |
| **Hải** | Trả lời **A4(a)** (công thức TF-IDF) |
| **Hải** | **A6:** Phân tích lỗi fit vectorizer trên test + data leakage |
| **Chung** | Bình review A4/A6 của Hải; Hải review `preprocess_text` của Bình |

**Done khi:** `text_clean` có trong DataFrame; Phần A đủ code + câu trả lời lý thuyết.

---

### 📅 Ngày 3 — Phần B (Trực quan hóa) + C1 (Chia dữ liệu)

**Mục tiêu:** Xong toàn bộ biểu đồ B; train/val/test 70/10/20; TF-IDF fit trên train.

| Dev | Tasks (~50%) |
|-----|----------------|
| **Bình** | **B1:** Bar chart + pie chart phân phối nhãn (tiêu đề, nhãn trục, màu, chú thích) + nhận xét |
| **Bình** | **B2:** Box plot độ dài văn bản theo nhãn + nhận xét |
| **Hải** | **B4:** Word cloud riêng 3 nhãn (nền tương phản, kích thước figure phù hợp) + nhận xét từ vựng |
| **Hải** | **C1:** Split 70/10/20, `stratify=y`, `random_state=42`; in phân phối từng tập |
| **Hải** | Giải thích markdown: tại sao `stratify`? val khác test thế nào? |
| **Hải** | `TfidfVectorizer` fit **chỉ trên train**, transform val/test |
| **Chung** | Thống nhất style biểu đồ (font, palette); kiểm tra không leakage |

**Done khi:** Phần B đủ 4 loại biểu đồ; `X_train`, `X_val`, `X_test` (đã vectorize) sẵn sàng.

---

### 📅 Ngày 4 — Phần C (Mô hình + Cross-Validation)

**Mục tiêu:** 4 mô hình có best hyperparameter trên val; Stratified 5-Fold CV.

| Dev | Tasks (~50%) |
|-----|----------------|
| **Bình** | **C2:** MultinomialNB (`alpha` {0.1, 0.5, 1.0}) + LogisticRegression (`C` {0.1, 1, 10}) |
| **Bình** | Trả lời **C2:** tham số `C` của Logistic Regression — overfitting |
| **Bình** | **C3:** StratifiedKFold k=5 cho NB + LR → mean ± std accuracy |
| **Hải** | **C2:** LinearSVC (`C` {0.1, 1, 10}) + RandomForest (`n_estimators`, `max_depth` grid đề bài) |
| **Hải** | Trả lời **C2:** margin/support vectors (SVM); RF vs overfitting |
| **Hải** | **C3:** StratifiedKFold k=5 cho SVM + RF → mean ± std accuracy |
| **Chung** | Gộp bảng CV 4 models; trả lời 3 câu **C3** (stratified vs k-fold, std cao nhất, CV vs val) |

**Done khi:** Best params từng model trên val; bảng CV đủ 4 dòng.

---

### 📅 Ngày 5 — Phần D + E + Báo cáo & Nộp bài

**Mục tiêu:** Đánh giá test, tuning, cải tiến, `report.pdf`, đóng gói `.zip`.

**Sáng — Đánh giá & tối ưu**

| Dev | Tasks (~50%) |
|-----|----------------|
| **Bình** | **D3:** 10 mẫu misclassified + phân tích đặc điểm chung + đề xuất xử lý |
| **Bình** | **D4:** Phản biện "Accuracy 85% đủ production?" (300–500 từ) |
| **Bình** | **E4:** Phản tư 300–400 từ (hướng cải tiến + bài học) |
| **Hải** | **D1:** Bảng test — Accuracy, Precision, Recall, F1 (macro) cho 4 models |
| **Hải** | **D2:** `classification_report` model tốt nhất; Precision/Recall theo nhãn; tình huống lọc review tiêu cực |
| **Hải** | **E1:** GridSearchCV / RandomizedSearchCV ≥2 model tốt nhất (≥3 params × ≥3 values) |
| **Hải** | **E2** + **E3:** Câu hỏi tuning; 1 kỹ thuật nâng cao (`class_weight` / SMOTE / …) + so sánh trước/sau |

**Chiều — Chung (cả hai)**

| Tasks |
|-------|
| Gộp `report.pdf` (Bình: EDA + viz + kết luận · Hải: methodology + models + bảng kết quả) |
| `Restart & Run All` notebook — đảm bảo có output |
| Rà checklist mục 5 |
| Đóng gói `Ten_nhom.zip` |
| Mock vấn đáp 30 phút — mỗi người giải thích 2 phần do đối phương làm |

**Done khi:** Zip sẵn sàng nộp; cả hai giải thích được toàn pipeline A→E.

---

## 5. Checklist nộp bài (rà soát cuối)

### Code & Notebook
- [ ] Dataset ≥ 5.000 mẫu, ≥ 3 nhãn
- [ ] `preprocess_text()` đủ 6 bước (lowercase, HTML, URL, special chars, tokenize, stopwords, stem/lemma)
- [ ] CountVectorizer & TfidfVectorizer đúng tham số đề bài
- [ ] Vectorizer **fit trên train**, transform val/test (không leakage)
- [ ] Split 70/10/20 + stratify + random_state cố định
- [ ] 4 mô hình đủ hyperparameter grid
- [ ] Stratified K-Fold k=5 + mean/std
- [ ] Bảng test: Accuracy, Precision, Recall, F1 (macro)
- [ ] GridSearch/RandomSearch ≥ 2 models
- [ ] ≥ 1 kỹ thuật cải tiến (E3)
- [ ] Notebook **Restart & Run All** thành công

### Trực quan hóa
- [ ] Bar + pie chart nhãn
- [ ] Box plot độ dài theo nhãn
- [ ] Word cloud ≥ 3 nhãn

### Câu hỏi lý thuyết (bắt buộc có lập luận)
- [ ] A3(a)(b), A4(a), A6
- [ ] C1 (stratify, val vs test), C2 (3 mô hình), C3 (3 câu)
- [ ] D2, D3, D4
- [ ] E2, E4

### Báo cáo
- [ ] `report.pdf` trình bày rõ, có bảng/biểu đồ chính
- [ ] Phản tư E4 (300–400 từ)

---

## 6. Cấu trúc notebook gợi ý

```
1. Giới thiệu & cài đặt thư viện
2. Load dữ liệu
── PHẦN A ──
3. A1 — EDA cơ bản
4. A2 — Thống kê độ dài văn bản
5. A3 — preprocess_text() + câu hỏi
6. A4 — BoW vs TF-IDF + câu hỏi
7. A6 — Phân tích lỗi vectorizer
── PHẦN B ──
8. B1 — Bar & Pie chart
9. B2 — Box plot
10. B4 — Word clouds
── PHẦN C ──
11. C1 — Train/Val/Test split
12. C2 — 4 models + câu hỏi
13. C3 — Stratified K-Fold CV
── PHẦN D ──
14. D1 — Bảng metrics test
15. D2 — Classification report + phân tích
16. D3 — 10 mẫu sai
17. D4 — Production debate
── PHẦN E ──
18. E1 — Hyperparameter tuning
19. E2 — Câu hỏi tuning
20. E3 — Kỹ thuật cải tiến
21. E4 — Phản tư & đề xuất
22. Kết luận
```

---

## 7. `requirements.txt` gợi ý

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
wordcloud>=1.9
nltk>=3.8
jupyter>=1.0
imbalanced-learn>=0.11   # nếu dùng SMOTE (E3)
```

Sau cài NLTK, chạy một lần:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')  # nếu dùng lemmatization
```

---

## 8. Rủi ro & mitigations

| Rủi ro | Cách xử lý |
|--------|------------|
| Dataset imbalanced mạnh | Ghi nhận ở A1; dùng stratify; cân nhắc `class_weight` hoặc SMOTE ở E3 |
| Notebook chạy lâu (RF, GridSearch) | Dùng subset khi debug; `n_jobs=-1`; RandomizedSearchCV nếu grid quá lớn |
| Tiếng Việt: stopwords/stemming kém | Dùng spaCy `vi_core_news_sm` hoặc bộ stopwords tiếng Việt custom |
| Data leakage | Luôn fit preprocessor/vectorizer trên train; Pipeline sklearn |
| Copy code không hiểu | Pair review hàng ngày; mock vấn đáp ngày 5 |

---

## 9. Tiêu chí chấm điểm (nhắc nhanh)

| Hạng mục | Điểm |
|----------|------|
| Phần A — Tiền xử lý | 20 |
| Phần B — Trực quan hóa | 20 |
| Phần C — Mô hình | 25 |
| Phần D — Đánh giá | 20 |
| Phần E — Tối ưu | 15 |
| **Chất lượng:** Code 20 + Phân tích 30 + Thực nghiệm 30 + Báo cáo 20 | (rubrics riêng) |

---

## 10. Bước tiếp theo (Ngày 2)

1. **Bình:** A2 + A3 + câu hỏi A3(a)(b)  
2. **Hải:** A4 + A4(a) + A6  
3. **Cả hai:** Standup 15 phút — review code buổi chiều trước khi merge notebook  

---

*Tài liệu này là roadmap nội bộ nhóm (5 ngày). Cập nhật tiến độ bằng cách tick checklist mục 5 và cột trạng thái mục 4.*
