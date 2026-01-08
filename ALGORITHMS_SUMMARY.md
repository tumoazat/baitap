# 📚 TÓM TẮT CÁC THUẬT TOÁN ĐÃ SỬ DỤNG

## 🎯 OVERVIEW

Project này sử dụng **4 thuật toán Machine Learning** và **1 phương pháp xử lý outliers** chính.

---

## 1️⃣ TIỀN XỬ LÝ DỮ LIỆU

### IQR Method (Interquartile Range)
**Mục đích**: Phát hiện và loại bỏ outliers (giá trị ngoại lai)

**Công thức**:
```
Q1 = Quartile thứ 1 (25%)
Q3 = Quartile thứ 3 (75%)
IQR = Q3 - Q1

Lower = Q1 - 1.5 × IQR
Upper = Q3 + 1.5 × IQR

Outliers = values < Lower OR values > Upper
```

**Ưu điểm**: 
- ✅ Robust, không cần phân phối chuẩn
- ✅ Dễ hiểu và implement

**Vị trí**: `notebooks/01_data_preprocessing.ipynb` - Cell "Xử Lý Giá Trị Ngoại Lai"

---

## 2️⃣ THUẬT TOÁN HỌC MÁY

### 🥉 Linear Regression
**Loại**: Simple regression
**Công thức**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

**Khi dùng**: Baseline model, dữ liệu tuyến tính
**Performance**: Thấp nhất nhưng nhanh và dễ hiểu

---

### 🥈 Random Forest
**Loại**: Ensemble (Bagging)
**Cách hoạt động**: 
1. Tạo nhiều decision trees từ random subsets
2. Mỗi tree vote
3. Kết quả = average của tất cả trees

**Hyperparameters**:
- `n_estimators=100`: Số trees
- `max_depth=20`: Độ sâu
- `min_samples_split=5`: Min để split

**Khi dùng**: Dữ liệu phức tạp, cần robust model
**Performance**: Cao, ít overfitting

---

### 🥇 XGBoost
**Loại**: Ensemble (Gradient Boosting)
**Cách hoạt động**:
1. Build trees tuần tự
2. Mỗi tree sửa lỗi của tree trước
3. Gradient descent + Regularization

**Hyperparameters**:
- `n_estimators=100`: Số boosting rounds
- `max_depth=7`: Độ sâu (nhỏ hơn RF)
- `learning_rate=0.1`: Tốc độ học

**Khi dùng**: Cần accuracy cao nhất
**Performance**: Rất cao, thường win competitions

---

### ⚡ LightGBM
**Loại**: Ensemble (Fast Gradient Boosting)
**Innovations**:
- GOSS: Sampling thông minh
- EFB: Bundle features
- Leaf-wise growth (vs level-wise)

**Hyperparameters**: Tương tự XGBoost

**Khi dùng**: Large datasets, cần speed
**Performance**: Tương đương XGBoost nhưng nhanh hơn 2-10x

---

## 3️⃣ PHƯƠNG PHÁP ĐÁNH GIÁ

### Train-Test Split
- **Training**: 80% - Model học từ data này
- **Testing**: 20% - Đánh giá trên unseen data

### K-Fold Cross-Validation
- Chia data thành K folds (K=5)
- Mỗi fold làm validation 1 lần
- Average của K scores → robust evaluation

### Grid Search
- Thử tất cả combinations của hyperparameters
- Chọn combination tốt nhất dựa trên CV score

---

## 4️⃣ METRICS (CHỈ SỐ ĐÁNH GIÁ)

| Metric | Công Thức | Range | Ý Nghĩa |
|--------|-----------|-------|---------|
| **MAE** | `(1/n)Σ\|y-ŷ\|` | [0,∞) | Sai số TB, VNĐ |
| **RMSE** | `√MSE` | [0,∞) | Penalize outliers |
| **R²** | `1-(SSres/SStot)` | (-∞,1] | % variance giải thích |
| **MAPE** | `(100/n)Σ\|(y-ŷ)/y\|` | [0,∞) | Sai số %, dễ hiểu |

---

## 5️⃣ SO SÁNH NHANH

| Model | Speed | Accuracy | Complexity | Overfitting Risk |
|-------|-------|----------|------------|------------------|
| Linear Reg | ⚡⚡⚡⚡⚡ | ⭐⭐ | Low | Low |
| Random Forest | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium | Medium |
| XGBoost | ⚡⚡ | ⭐⭐⭐⭐⭐ | High | Medium-High |
| LightGBM | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | High | Medium-High |

---

## 📂 VỊ TRÍ TRONG CODE

### Preprocessing
- **File**: `src/preprocessing.py`
- **Notebook**: `notebooks/01_data_preprocessing.ipynb`
- **Thuật toán**: IQR outlier detection, Label Encoding

### Model Training
- **File**: `src/model.py`
- **Notebook**: `notebooks/02_model_training.ipynb`
- **Thuật toán**: Linear Regression, Random Forest, XGBoost, LightGBM

---

## 🚀 WORKFLOW TỔNG QUÁT

```
DATA
  ↓
[1] Load & EDA
  ↓
[2] Clean (remove duplicates, handle missing)
  ↓
[3] Remove Outliers (IQR Method) ← THUẬT TOÁN 1
  ↓
[4] Encode Categorical (Label Encoding)
  ↓
[5] Train-Test Split (80/20)
  ↓
[6] Train 4 Models:
    • Linear Regression      ← THUẬT TOÁN 2
    • Random Forest          ← THUẬT TOÁN 3
    • XGBoost                ← THUẬT TOÁN 4
    • LightGBM               ← THUẬT TOÁN 5
  ↓
[7] Evaluate with Metrics (MAE, RMSE, R², MAPE)
  ↓
[8] Cross-Validation
  ↓
[9] Select Best Model
  ↓
[10] Deploy (Save model)
```

---

## 📖 TÀI LIỆU CHI TIẾT

Xem file `GIAI_THICH_THUAT_TOAN.md` để có:
- ✅ Giải thích chi tiết từng thuật toán
- ✅ Công thức toán học đầy đủ
- ✅ Ví dụ cụ thể
- ✅ Ưu/nhược điểm
- ✅ Best practices
- ✅ Tips để tránh overfitting

---

## ✨ KEY TAKEAWAYS

1. **IQR Method**: Loại bỏ outliers robust, không cần normal distribution
2. **Linear Regression**: Simple baseline, dễ hiểu
3. **Random Forest**: Robust ensemble, ít overfitting
4. **XGBoost**: Highest accuracy với proper tuning
5. **LightGBM**: Fastest training, tốt cho big data
6. **Cross-Validation**: Essential để đánh giá reliable
7. **Multiple Metrics**: Dùng nhiều metrics để hiểu model toàn diện

---

**💡 Tip**: Tất cả các cell trong notebooks đều có comment chi tiết giải thích từng bước!
