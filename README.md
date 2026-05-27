# Bài Tập Lớn Cuối Kỳ — Phân tích Văn bản & Phân loại Tự động (ML)

**Thời gian thực hiện (dự kiến):** 1 tuần (7 ngày)  
**Nhóm:** Bình + Hải  
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

| Tiêu chí | Yêu cầu |
|----------|---------|
| Ngôn ngữ | Tiếng Anh **hoặc** Tiếng Việt |
| Số mẫu | ≥ 5.000 |
| Số nhãn | ≥ 3 nhãn phân loại |
| Gợi ý nguồn | Amazon Reviews (Kaggle), Shopee/Tiki, IMDB, hoặc web scraping (cần mô tả phương pháp) |

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

| Vai trò | Dev | Trách nhiệm chính |
|---------|-----|-------------------|
| **Dev A — Data & NLP** | **Bình** | Dataset, EDA (A1–A2), `preprocess_text` (A3), vectorizers (A4), trực quan hóa (B), trả lời câu hỏi A & B trong notebook |
| **Dev B — ML & Evaluation** | **Hải** | Train/val/test split (C1), 4 mô hình + CV (C2–C3), đánh giá test (D), tuning & cải tiến (E), trả lời câu hỏi C, D, E |
| **Chung** | Cả hai | Review code lẫn nhau, viết `report.pdf`, kiểm tra output notebook, đóng gói `.zip`, chuẩn bị vấn đáp |

> **Quy tắc làm việc:** Mỗi ngày sync 15 phút (standup); merge notebook qua Git hoặc chia section rõ ràng; **cả hai phải hiểu và giải thích được mọi dòng code**.

---

## 4. Lộ trình 1 tuần (7 ngày)

Giả sử **Ngày 1 = Thứ Hai**. Điều chỉnh ngày thực tế theo lịch nhóm.

---

### 📅 Ngày 1 — Thiết lập & Thu thập dữ liệu

**Mục tiêu:** Có dataset sạch cơ bản, repo và notebook khung sẵn sàng.

| Dev | Tasks |
|-----|-------|
| **Bình** | Chọn & tải dataset (≥5k mẫu, ≥3 nhãn); tạo cấu trúc thư mục `data/`; khám phá sơ bộ (head, info, value_counts); ghi chú nguồn dữ liệu |
| **Hải** | Khởi tạo repo, `requirements.txt`, notebook skeleton (mục lục Phần A→E); thiết lập `random_state=42` |
| **Chung** | Thống nhất tên cột (`text`, `label`), ngôn ngữ xử lý (EN/VI), thư viện NLP (NLTK vs spaCy) |

**Done khi:** Data load được, ≥5000 rows, ≥3 labels, notebook có outline.

---

### 📅 Ngày 2 — EDA & Tiền xử lý (Phần A)

**Mục tiêu:** Hoàn thành A1, A2, A3, bắt đầu A4.

| Dev | Tasks |
|-----|-------|
| **Bình** | **A1:** 10 dòng đầu, describe, dtype, phân phối nhãn, nhận xét imbalanced |
| **Bình** | **A2:** Tính word_count, char_count; thống kê theo nhãn; nhận xét tương quan |
| **Bình** | **A3:** Viết `preprocess_text()` đầy đủ pipeline; chọn lemmatization **hoặc** stemming + giải thích |
| **Bình** | Trả lời A3(a)(b), A6 trong markdown |
| **Hải** | Review hàm preprocess; chuẩn bị pipeline sklearn (Pipeline + TF-IDF) cho ngày 3 |
| **Hải** | **A4 (bắt đầu):** CountVectorizer & TfidfVectorizer fit mẫu, so sánh shape/sparsity |

**Done khi:** Cột `text_clean` có sẵn; câu trả lời A1–A3, A6 trong notebook.

---

### 📅 Ngày 3 — Feature extraction & Trực quan hóa (A4 + Phần B)

**Mục tiêu:** Xong A4; hoàn thành B1, B2, B4.

| Dev | Tasks |
|-----|-------|
| **Bình** | **A4:** Hoàn thiện BoW vs TF-IDF; giải thích công thức TF-IDF |
| **Bình** | **B1:** Bar + pie chart phân phối nhãn |
| **Bình** | **B2:** Box plot độ dài văn bản theo nhãn |
| **Bình** | **B4:** Word cloud từng nhãn (≥3) |
| **Hải** | **C1:** Split 70/10/20 stratified; in phân phối train/val/test; viết giải thích stratify |
| **Hải** | Xây `TfidfVectorizer(max_features=10000, ngram_range=(1,2))` trên **train only** |

**Done khi:** Tất cả biểu đồ Phần B có trong notebook; train/val/test đã tách.

---

### 📅 Ngày 4 — Huấn luyện 4 mô hình (Phần C — C2)

**Mục tiêu:** Fit 4 mô hình với hyperparameter grid gợi ý; chọn best trên validation.

| Dev | Tasks |
|-----|-------|
| **Hải** | MultinomialNB — grid alpha {0.1, 0.5, 1.0} |
| **Hải** | LogisticRegression — C {0.1, 1, 10}, solver='lbfgs', max_iter=1000 |
| **Hải** | LinearSVC — C {0.1, 1, 10} |
| **Hải** | RandomForest — n_estimators {100,200}, max_depth {None,10,20} |
| **Hải** | Trả lời câu hỏi C2 (C, margin, RF overfitting) |
| **Bình** | Review pipeline: vectorizer fit trên train, transform val/test |
| **Bình** | Bắt đầu draft report — mục Giới thiệu + Phần A & B |

**Done khi:** 4 mô hình có best params trên val; accuracy val của từng model.

---

### 📅 Ngày 5 — Cross-Validation & Đánh giá Test (C3 + Phần D)

**Mục tiêu:** Stratified 5-Fold CV; bảng metrics trên test; phân tích sâu.

| Dev | Tasks |
|-----|-------|
| **Hải** | **C3:** StratifiedKFold k=5, mean ± std accuracy 4 models |
| **Hải** | Trả lời 3 câu C3 (stratified vs k-fold, std cao nhất, CV vs val) |
| **Hải** | **D1:** Bảng Accuracy, Precision, Recall, F1 (macro) trên **test** |
| **Hải** | **D2:** classification_report model tốt nhất; phân tích nhãn tiêu cực |
| **Bình** | **D3:** Lấy 10 mẫu misclassified; phân tích đặc điểm + đề xuất cải thiện |
| **Bình** | **D4:** Viết phản biện "85% accuracy đủ production?" |
| **Bình** | Cập nhật report Phần C & D |

**Done khi:** Bảng D1 điền số thực; D2–D4 có lập luận rõ ràng.

---

### 📅 Ngày 6 — Tối ưu hóa & Cải tiến (Phần E)

**Mục tiêu:** GridSearch/RandomSearch ≥2 models; ≥1 kỹ thuật nâng cao; so sánh trước/sau.

| Dev | Tasks |
|-----|-------|
| **Hải** | **E1:** GridSearchCV hoặc RandomizedSearchCV cho 2 model tốt nhất (≥3 params × ≥3 values) |
| **Hải** | Ghi best params + so sánh baseline vs tuned |
| **Hải** | **E3:** Chọn & triển khai 1 kỹ thuật (gợi ý: `class_weight='balanced'` hoặc SMOTE nếu imbalanced mạnh) |
| **Hải** | Trả lời E2 (Grid vs Random, CV trong tuning, diminishing returns) |
| **Bình** | **E4:** Viết phản tư 300–400 từ (BERT/embeddings, thêm data, bài học) |
| **Bình** | Hoàn thiện report Phần E |
| **Chung** | Chạy lại toàn bộ notebook top-to-bottom (`Restart & Run All`) |

**Done khi:** E1–E4 có số liệu và phân tích; notebook chạy không lỗi.

---

### 📅 Ngày 7 — Báo cáo, QA & Đóng gói

**Mục tiêu:** Nộp bài đúng format; sẵn sàng vấn đáp.

| Dev | Tasks |
|-----|-------|
| **Bình** | Hoàn thiện `report.pdf`: cover, mục lục, EDA, visualization, kết luận |
| **Hải** | Hoàn thiện `report.pdf`: methodology, models, results tables, tuning |
| **Chung** | Kiểm tra checklist đề bài (mục 5 bên dưới) |
| **Chung** | Export notebook có **đầy đủ output** |
| **Chung** | Đóng gói `Ten_nhom.zip` (notebook + report + data/script) |
| **Chung** | Mock vấn đáp 30 phút: mỗi người giải thích 2 phần bất kỳ |

**Done khi:** File zip sẵn sàng nộp; cả hai hiểu toàn bộ pipeline.

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
| Copy code không hiểu | Pair review hàng ngày; mock vấn đáp ngày 7 |

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

## 10. Bước tiếp theo ngay hôm nay

1. **Bình:** Chọn dataset và đặt vào `data/`  
2. **Hải:** Tạo `notebook.ipynb` + `requirements.txt` theo outline mục 6  
3. **Cả hai:** Họp 15 phút thống nhất dataset và ngôn ngữ (EN/VI)  

---

*Tài liệu này là roadmap nội bộ nhóm. Cập nhật tiến độ hàng ngày bằng cách tick checklist mục 5.*
