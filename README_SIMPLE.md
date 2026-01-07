# 🎓 Dự Án Dự Đoán Giá Nhà - Dành cho Sinh Viên Mới Học Machine Learning

> **Phiên bản đơn giản** - Dễ hiểu, dễ học, dễ thực hành!

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Mục Tiêu Dự Án

Dự án này được thiết kế đặc biệt cho **sinh viên mới học Machine Learning**, giúp bạn:

✅ Hiểu rõ quy trình ML từ đầu đến cuối  
✅ Học cách xử lý dữ liệu thực tế  
✅ Hiểu 2 thuật toán cơ bản: Linear Regression & Random Forest  
✅ Xây dựng ứng dụng web đơn giản với Streamlit  
✅ Có thể giải thích code cho người khác  

## 📚 Bạn Sẽ Học Được Gì?

### 1. Xử Lý Dữ Liệu (Preprocessing)
- Đọc file CSV
- Làm sạch dữ liệu (loại bỏ ký tự đặc biệt, giá trị thiếu)
- Xử lý outliers (giá trị bất thường)
- Chuyển đổi dữ liệu về dạng số

### 2. Huấn Luyện Model
- Linear Regression (thuật toán đơn giản nhất)
- Random Forest (mạnh hơn một chút)
- Chia dữ liệu train/test
- Đánh giá model bằng MAE, RMSE, R²

### 3. Xây Dựng Web App
- Tạo giao diện với Streamlit
- Nhận input từ người dùng
- Hiển thị kết quả dự đoán
- Vẽ biểu đồ phân tích

## 🚀 Bắt Đầu Nhanh

### Bước 1: Cài Đặt

```bash
# Clone project
git clone https://github.com/tumoazat/baitap.git
cd baitap

# Tạo môi trường ảo (recommended)
python -m venv .venv

# Kích hoạt môi trường
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 2: Tải Dữ Liệu

```bash
# Chạy script tự động tải
python download_dataset.py

# Hoặc tải thủ công từ:
# https://www.kaggle.com/datasets/ladcva/vietnam-housing-dataset-hanoi/data
```

### Bước 3: Xử Lý Dữ Liệu

```bash
# Chạy script xử lý dữ liệu
python src/preprocessing_simple.py
```

### Bước 4: Huấn Luyện Model

```bash
# Chạy script huấn luyện
python src/model_simple.py
```

### Bước 5: Chạy Web App

```bash
# Chạy phiên bản đơn giản
streamlit run app/streamlit_app_simple.py

# Hoặc phiên bản đầy đủ
streamlit run app/streamlit_app.py
```

## 📁 Cấu Trúc Dự Án (Đơn Giản)

```
baitap/
│
├── app/
│   ├── streamlit_app_simple.py      # Web app đơn giản ⭐
│   └── streamlit_app.py             # Web app đầy đủ
│
├── src/
│   ├── preprocessing_simple.py      # Xử lý dữ liệu ⭐
│   ├── model_simple.py              # Huấn luyện model ⭐
│   ├── preprocessing.py             # (Phiên bản nâng cao)
│   └── model.py                     # (Phiên bản nâng cao)
│
├── data/
│   ├── vietnam_housing_dataset.csv  # Dữ liệu gốc
│   └── processed_housing_data.csv   # Dữ liệu đã xử lý
│
├── models/
│   └── simple_housing_model.pkl     # Model đã train
│
├── notebooks/
│   └── beginner_tutorial.ipynb      # Notebook hướng dẫn ⭐
│
├── requirements.txt                  # Danh sách thư viện
├── download_dataset.py               # Script tải dữ liệu
└── README.md                         # File này
```

**⭐ = Files quan trọng dành cho sinh viên**

## 💡 Hướng Dẫn Chi Tiết Cho Từng File

### 1. `preprocessing_simple.py` - Xử Lý Dữ Liệu

**Mục đích:** Làm sạch dữ liệu thô để có thể train model

```python
# Cách sử dụng
from src.preprocessing_simple import HousingDataPreprocessor

# Khởi tạo
processor = HousingDataPreprocessor('data/vietnam_housing_dataset.csv')

# Xử lý toàn bộ (1 dòng code!)
df_clean = processor.preprocess_all()

# Lưu kết quả
processor.save_processed_data('data/processed_housing_data.csv')
```

**Các bước trong file:**
1. Đọc dữ liệu từ CSV
2. Làm sạch các cột số (loại bỏ text, ký tự đặc biệt)
3. Tính giá nhà từ Diện tích × Giá/m²
4. Loại bỏ outliers (nhà quá rẻ/đắt, quá nhỏ/lớn)
5. Xóa dòng thiếu thông tin quan trọng

### 2. `model_simple.py` - Huấn Luyện Model

**Mục đích:** Train model ML để dự đoán giá nhà

```python
# Cách sử dụng
from src.model_simple import SimpleHousingModel

# Khởi tạo
model_trainer = SimpleHousingModel()

# Đọc dữ liệu đã xử lý
model_trainer.load_data('data/processed_housing_data.csv')

# Chuẩn bị features
X, y = model_trainer.prepare_features()

# Chia train/test
model_trainer.split_data(X, y)

# Train models
model_trainer.train_linear_regression()
model_trainer.train_random_forest()

# So sánh kết quả
model_trainer.compare_models()

# Lưu model tốt nhất
model_trainer.save_model('Random Forest', 'models/best_model.pkl')
```

**Các bước trong file:**
1. Đọc dữ liệu đã xử lý
2. Chọn features (Diện tích, Số tầng)
3. Chia train/test (80/20)
4. Train Linear Regression
5. Train Random Forest
6. Đánh giá cả 2 models
7. Lưu model tốt nhất

### 3. `streamlit_app_simple.py` - Web App

**Mục đích:** Tạo giao diện web để dự đoán giá nhà

```bash
# Chạy app
streamlit run app/streamlit_app_simple.py
```

**Tính năng:**
- Tab 1: Dự đoán giá nhà (nhập thông tin → xem kết quả)
- Tab 2: Phân tích dữ liệu (biểu đồ, thống kê)
- Sidebar: Hướng dẫn sử dụng

## 📊 Hiểu Về Dữ Liệu

### Dataset
- **Nguồn:** Kaggle - Vietnam Housing Dataset
- **Số lượng:** ~82,000 bản ghi
- **Khu vực:** Hà Nội, Việt Nam

### Các Cột Quan Trọng

| Cột | Ý nghĩa | Ví dụ |
|-----|---------|-------|
| Quận | Khu vực | "Đống Đa", "Cầu Giấy" |
| Diện tích | Diện tích đất/sàn | 50 m² |
| Số tầng | Số tầng nhà | 3 tầng |
| Giá/m² | Giá mỗi m² | 86.96 triệu/m² |
| **Giá** | **Giá tổng (target)** | **4.5 tỷ VNĐ** |

### Ví Dụ Một Dòng Dữ Liệu

```
Quận: Cầu Giấy
Diện tích: 50 m²
Số tầng: 4
Giá/m²: 86.96 triệu
→ Giá = 50 × 86.96 × 1,000,000 = 4,348,000,000 VNĐ (≈ 4.35 tỷ)
```

## 🤖 Hiểu Về Machine Learning

### Linear Regression (Hồi quy Tuyến tính)

**Ý tưởng:** Tìm một đường thẳng/mặt phẳng fit với dữ liệu

```
Giá = a × Diện_tích + b × Số_tầng + c
```

**Ưu điểm:**
- ✅ Đơn giản, dễ hiểu
- ✅ Nhanh
- ✅ Có thể giải thích được

**Nhược điểm:**
- ❌ Chỉ fit với mối quan hệ tuyến tính
- ❌ Kém chính xác với dữ liệu phức tạp

### Random Forest (Rừng Cây Quyết định)

**Ý tưởng:** Tạo nhiều "cây quyết định" và lấy trung bình kết quả

```
Nếu Diện_tích > 100m²:
    Nếu Quận == "Đống Đa":
        Giá ≈ 10 tỷ
    Ngược lại:
        Giá ≈ 8 tỷ
Ngược lại:
    Giá ≈ 5 tỷ
```

**Ưu điểm:**
- ✅ Chính xác hơn Linear Regression
- ✅ Xử lý được mối quan hệ phức tạp
- ✅ Không cần chuẩn hóa dữ liệu

**Nhược điểm:**
- ❌ Chậm hơn
- ❌ Khó giải thích

## 📈 Đánh Giá Model

### Các Chỉ Số

| Chỉ số | Ý nghĩa | Mục tiêu |
|--------|---------|----------|
| **MAE** | Sai số trung bình | Càng nhỏ càng tốt |
| **RMSE** | Sai số (phạt nặng lỗi lớn) | Càng nhỏ càng tốt |
| **R²** | Model giải thích được bao nhiêu % | Càng gần 1 càng tốt |
| **MAPE** | Sai số theo % | Càng nhỏ càng tốt |

### Ví Dụ Kết Quả

```
Linear Regression:
- MAE:  0.85 tỷ    (sai trung bình 850 triệu)
- R²:   0.75       (giải thích được 75% biến động giá)

Random Forest:
- MAE:  0.52 tỷ    (sai trung bình 520 triệu) ← Tốt hơn!
- R²:   0.89       (giải thích được 89% biến động giá) ← Tốt hơn!

→ Random Forest là model tốt hơn!
```

## 🎓 Lộ Trình Học Cho Sinh Viên

### Tuần 1-2: Làm Quen Với Dữ Liệu
1. ✅ Chạy `preprocessing_simple.py`
2. ✅ Đọc và hiểu từng dòng code
3. ✅ Thử thay đổi ngưỡng outliers và xem kết quả
4. ✅ Xem file CSV trước và sau xử lý

### Tuần 3-4: Học Machine Learning
1. ✅ Chạy `model_simple.py`
2. ✅ So sánh Linear Regression vs Random Forest
3. ✅ Thử thay đổi số cây trong Random Forest
4. ✅ Dự đoán giá cho một vài ngôi nhà

### Tuần 5-6: Xây Dựng Web App
1. ✅ Chạy `streamlit_app_simple.py`
2. ✅ Thử dự đoán giá với nhiều input khác nhau
3. ✅ Xem các biểu đồ phân tích
4. ✅ Thử thêm features mới vào model

### Tuần 7-8: Nâng Cao
1. ✅ Chạy phiên bản đầy đủ (`streamlit_app.py`)
2. ✅ Học thêm về XGBoost, LightGBM
3. ✅ Thử thêm features khác (Loại nhà, Giấy tờ, v.v.)
4. ✅ Cải thiện độ chính xác model

## 💻 Các Lệnh Hữu Ích

```bash
# Xử lý dữ liệu
python src/preprocessing_simple.py

# Train model
python src/model_simple.py

# Chạy web app đơn giản
streamlit run app/streamlit_app_simple.py

# Chạy web app đầy đủ
streamlit run app/streamlit_app.py

# Xem thống kê dữ liệu
python -c "import pandas as pd; df = pd.read_csv('data/processed_housing_data.csv'); print(df.describe())"

# Kiểm tra model đã lưu
python -c "import joblib; model = joblib.load('models/simple_housing_model.pkl'); print(type(model))"
```

## ❓ Câu Hỏi Thường Gặp

### 1. Tại sao model dự đoán không chính xác 100%?

Vì giá nhà phụ thuộc vào **rất nhiều yếu tố**:
- Vị trí cụ thể (gần chợ, trường học, bệnh viện?)
- Tình trạng nhà (mới/cũ, đã sửa chữa?)
- Thời điểm (thị trường đang lên/xuống?)
- Yếu tố cá nhân của người mua/bán

Model chỉ biết **Diện tích + Số tầng**, nên không thể 100% chính xác.

### 2. Làm sao để cải thiện model?

- ✅ Thêm features (Loại nhà, Quận, Số phòng ngủ, v.v.)
- ✅ Thu thập thêm dữ liệu
- ✅ Dùng thuật toán mạnh hơn (XGBoost, Neural Networks)
- ✅ Điều chỉnh hyperparameters

### 3. Tại sao phải chia train/test?

Để kiểm tra xem model có **"học tủ"** (overfitting) không:
- Train tốt, Test tốt → Model tốt ✅
- Train tốt, Test kém → Model học tủ ❌

### 4. Linear Regression hay Random Forest?

| | Linear Regression | Random Forest |
|---|---|---|
| Độ chính xác | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Tốc độ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Dễ hiểu | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**Kết luận:** Bắt đầu với Linear Regression để học, sau đó dùng Random Forest cho độ chính xác cao hơn.

## 🐛 Gặp Lỗi?

### Lỗi: "ModuleNotFoundError"
```bash
# Cài lại thư viện
pip install -r requirements.txt
```

### Lỗi: "FileNotFoundError: Dataset not found"
```bash
# Tải dữ liệu
python download_dataset.py
```

### Lỗi: "KeyError: 'Giá'"
```bash
# Chạy lại preprocessing
python src/preprocessing_simple.py
```

## 📚 Tài Liệu Tham Khảo

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Machine Learning Crash Course (Google)](https://developers.google.com/machine-learning/crash-course)

## 🤝 Đóng Góp

Dự án này được tạo ra cho mục đích học tập. Nếu bạn tìm thấy lỗi hoặc có ý tưởng cải thiện, hãy:

1. Fork repo này
2. Tạo branch mới (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add some improvement'`)
4. Push (`git push origin feature/improvement`)
5. Tạo Pull Request

## 📝 License

MIT License - Tự do sử dụng cho mục đích học tập

## 👨‍🎓 Tác Giả

Made with ❤️ cho sinh viên Việt Nam đang học Machine Learning

---

**🎯 Mục tiêu cuối cùng:** Sau khi hoàn thành dự án này, bạn sẽ tự tin giải thích được:
- ML là gì và hoạt động như thế nào
- Cách xử lý dữ liệu thực tế
- Cách train và đánh giá model
- Cách deploy model thành web app

**Chúc bạn học tốt! 🚀**
