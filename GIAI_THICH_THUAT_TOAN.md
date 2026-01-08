# GIẢI THÍCH CÁC THUẬT TOÁN ĐÃ SỬ DỤNG

## 1. THUẬT TOÁN TIỀN XỬ LÝ DỮ LIỆU

### 1.1. Xử Lý Giá Trị Thiếu (Missing Values)
**Mô tả**: Xử lý các giá trị bị thiếu trong dataset theo quy tắc cụ thể
**Phương pháp**:
- **Categorical variables**: Điền với "Unknown" hoặc giá trị từ cột khác
- **Numerical variables**: Điền với giá trị mặc định (median, mean, hoặc 1)

**Ưu điểm**: Giữ lại toàn bộ dữ liệu, tránh mất thông tin
**Nhược điểm**: Có thể tạo bias nếu có quá nhiều giá trị thiếu

### 1.2. Phát Hiện và Loại Bỏ Outliers bằng IQR (Interquartile Range)
**Mô tả**: IQR là phương pháp thống kê để phát hiện giá trị ngoại lai
**Công thức**:
```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```
Trong đó:
- Q1: Quartile thứ 1 (25th percentile)
- Q3: Quartile thứ 3 (75th percentile)
- IQR: Khoảng tứ phân vị

**Cách hoạt động**:
1. Tính Q1 (25% dữ liệu nhỏ hơn giá trị này)
2. Tính Q3 (75% dữ liệu nhỏ hơn giá trị này)
3. Tính IQR = Q3 - Q1
4. Xác định ngưỡng dưới và ngưỡng trên
5. Loại bỏ các giá trị nằm ngoài ngưỡng

**Ưu điểm**: 
- Không phụ thuộc vào phân phối chuẩn
- Robust với dữ liệu lệch (skewed)
- Dễ hiểu và implement

**Nhược điểm**: 
- Có thể loại bỏ quá nhiều dữ liệu nếu threshold quá nhỏ
- Không phù hợp với dữ liệu có nhiều peaks

### 1.3. Label Encoding
**Mô tả**: Chuyển đổi categorical variables thành số
**Cách hoạt động**: 
- Gán mỗi category một số nguyên duy nhất
- Ví dụ: ['Ba Đình', 'Hoàn Kiếm', 'Đống Đa'] → [0, 1, 2]

**Ưu điểm**: Đơn giản, tiết kiệm bộ nhớ
**Nhược điểm**: Tạo thứ tự giả (ordinal) giữa các category

---

## 2. THUẬT TOÁN HỌC MÁY (MACHINE LEARNING)

### 2.1. Linear Regression (Hồi Quy Tuyến Tính)
**Mô tả**: Mô hình đơn giản nhất, tìm mối quan hệ tuyến tính giữa features và target
**Công thức**: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

**Cách hoạt động**:
1. Khởi tạo các hệ số β (coefficients)
2. Tính dự đoán ŷ = β₀ + β₁x₁ + ... + βₙxₙ
3. Tính loss (sai số): MSE = Σ(y - ŷ)²
4. Cập nhật β để minimize loss (sử dụng Ordinary Least Squares)

**Ưu điểm**: 
- Đơn giản, dễ hiểu và interpret
- Training nhanh
- Ít tham số, ít overfitting

**Nhược điểm**: 
- Chỉ capture được mối quan hệ tuyến tính
- Hiệu suất thấp với dữ liệu phức tạp
- Nhạy cảm với outliers

**Khi nào dùng**: Dữ liệu có mối quan hệ tuyến tính rõ ràng, cần model đơn giản và dễ giải thích

---

### 2.2. Random Forest (Rừng Ngẫu Nhiên)
**Mô tả**: Ensemble method kết hợp nhiều Decision Trees
**Cách hoạt động**:
1. **Bootstrap Aggregating (Bagging)**:
   - Tạo n subsets ngẫu nhiên từ training data (sampling with replacement)
   - Mỗi subset có cùng kích thước với original data

2. **Build Decision Trees**:
   - Train một Decision Tree trên mỗi subset
   - Tại mỗi split node, chỉ xét một subset ngẫu nhiên của features
   - Tree được grow đến maximum depth hoặc đến khi không thể split

3. **Prediction**:
   - Mỗi tree đưa ra dự đoán riêng
   - Kết quả cuối cùng = average của tất cả predictions (regression)

**Hyperparameters chính**:
- `n_estimators`: Số lượng trees (100-500)
- `max_depth`: Độ sâu tối đa của mỗi tree (10-30)
- `min_samples_split`: Số mẫu tối thiểu để split node (2-10)
- `min_samples_leaf`: Số mẫu tối thiểu ở leaf node (1-5)
- `max_features`: Số features xét tại mỗi split

**Ưu điểm**:
- Hiệu suất cao, robust
- Giảm overfitting so với single tree
- Handle được non-linear relationships
- Tính được feature importance
- Không cần feature scaling
- Handle được missing values tốt

**Nhược điểm**:
- Training chậm với dataset lớn
- Model size lớn (nhiều trees)
- Khó interpret so với single tree
- Có thể overfit với noisy data

**Khi nào dùng**: 
- Dữ liệu có relationships phức tạp
- Cần model robust và chính xác
- Có đủ computational resources

---

### 2.3. XGBoost (Extreme Gradient Boosting)
**Mô tả**: Advanced ensemble method sử dụng gradient boosting với nhiều tối ưu hóa
**Cách hoạt động**:

1. **Sequential Tree Building**:
   ```
   F₀(x) = initial prediction (mean)
   For m = 1 to M:
       1. Tính residuals: rᵢ = yᵢ - F_{m-1}(xᵢ)
       2. Fit tree hₘ(x) trên residuals
       3. Update: F_m(x) = F_{m-1}(x) + η × hₘ(x)
   ```

2. **Gradient Descent**:
   - Sử dụng gradient của loss function để tối ưu
   - Loss = Σ L(yᵢ, ŷᵢ) + Σ Ω(fₖ)
   - Trong đó Ω là regularization term

3. **Key Innovations**:
   - **Regularization**: L1 (Lasso) và L2 (Ridge) để prevent overfitting
   - **Tree Pruning**: Prune trees sau khi build (không phải pre-pruning)
   - **Parallel Processing**: Split finding được parallelize
   - **Cache Optimization**: Tối ưu memory và speed

**Hyperparameters chính**:
- `n_estimators`: Số lượng boosting rounds (100-1000)
- `max_depth`: Độ sâu của trees (3-10, thường nhỏ hơn Random Forest)
- `learning_rate` (η): Tốc độ học (0.01-0.3)
- `subsample`: Tỷ lệ samples dùng cho mỗi tree (0.5-1.0)
- `colsample_bytree`: Tỷ lệ features dùng cho mỗi tree (0.5-1.0)
- `gamma`: Minimum loss reduction để split (0-5)
- `reg_alpha`: L1 regularization (0-1)
- `reg_lambda`: L2 regularization (1-10)

**Ưu điểm**:
- Hiệu suất rất cao, thường win Kaggle competitions
- Handle missing values tự động
- Built-in regularization giảm overfitting
- Có thể parallel và distributed training
- Efficient với large datasets
- Flexible với custom loss functions

**Nhược điểm**:
- Dễ overfit nếu tune hyperparameters sai
- Nhiều hyperparameters cần tune
- Training phức tạp hơn Random Forest
- Cần nhiều experience để tune tốt

**Khi nào dùng**:
- Cần highest accuracy
- Có thời gian để tune hyperparameters
- Dữ liệu có structure phức tạp

---

### 2.4. LightGBM (Light Gradient Boosting Machine)
**Mô tả**: Variant của Gradient Boosting được Microsoft phát triển, focus vào speed và efficiency
**Cách hoạt động**:

1. **Gradient-based One-Side Sampling (GOSS)**:
   - Giữ lại tất cả instances với gradient lớn (well-classified)
   - Random sample các instances với gradient nhỏ
   - Giảm data size nhưng vẫn giữ accuracy

2. **Exclusive Feature Bundling (EFB)**:
   - Bundle các features hiếm khi cùng non-zero
   - Giảm số features cần xử lý
   - Tăng tốc training đáng kể

3. **Leaf-wise Tree Growth**:
   ```
   Level-wise (XGBoost):    Leaf-wise (LightGBM):
         Root                      Root
        /    \                    /    \
       A      B                  A      B
      / \    / \                / \
     C   D  E  F               C   E
   ```
   - Grow tree theo leaf có highest loss reduction
   - Deeper và more accurate nhưng dễ overfit hơn

**Hyperparameters chính**:
- `n_estimators`: Số boosting iterations (100-1000)
- `max_depth`: Độ sâu tối đa (-1 = no limit)
- `learning_rate`: Tốc độ học (0.01-0.3)
- `num_leaves`: Số leaves tối đa (31-256)
- `min_child_samples`: Số mẫu tối thiểu ở leaf (20-100)
- `subsample`: Tỷ lệ data sampling (0.5-1.0)
- `colsample_bytree`: Tỷ lệ feature sampling (0.5-1.0)
- `reg_alpha`: L1 regularization
- `reg_lambda`: L2 regularization

**Ưu điểm**:
- Training rất nhanh, nhanh hơn XGBoost 2-10x
- Memory efficient
- Hiệu suất cao, comparable với XGBoost
- Handle được large datasets và high-dimensional data
- Support categorical features trực tiếp
- Good accuracy với ít hyperparameter tuning

**Nhược điểm**:
- Dễ overfit với small datasets (< 10K samples)
- Leaf-wise growth có thể tạo deep trees
- Cần cẩn thận với num_leaves và max_depth
- Ít stable hơn XGBoost với noisy data

**Khi nào dùng**:
- Large datasets (> 10K samples)
- Cần training speed cao
- Limited memory
- High-dimensional data

---

## 3. PHƯƠNG PHÁP ĐÁNH GIÁ VÀ TỐI ƯU

### 3.1. Train-Test Split
**Mô tả**: Chia dữ liệu thành 2 phần riêng biệt
**Cách hoạt động**:
- Training set (80%): Dùng để train model
- Test set (20%): Dùng để đánh giá hiệu suất

**Mục đích**: Đánh giá model trên dữ liệu chưa từng thấy (unseen data)

---

### 3.2. Cross-Validation (K-Fold)
**Mô tả**: Phương pháp đánh giá robust hơn train-test split
**Cách hoạt động**:
```
Iteration 1: [Test][Train][Train][Train][Train]
Iteration 2: [Train][Test][Train][Train][Train]
Iteration 3: [Train][Train][Test][Train][Train]
Iteration 4: [Train][Train][Train][Test][Train]
Iteration 5: [Train][Train][Train][Train][Test]
```

**Quy trình**:
1. Chia data thành K folds (thường K=5 hoặc 10)
2. Với mỗi fold:
   - Dùng fold đó làm validation set
   - Dùng K-1 folds còn lại làm training set
   - Train và evaluate model
3. Tính average performance across K folds

**Ưu điểm**:
- Sử dụng toàn bộ data cho cả training và validation
- Đánh giá robust và reliable hơn
- Giảm variance trong evaluation

**Nhược điểm**:
- Training chậm hơn K lần
- Không phù hợp với very large datasets

---

### 3.3. Hyperparameter Tuning (Grid Search)
**Mô tả**: Tìm kiếm tổ hợp hyperparameters tốt nhất
**Cách hoạt động**:
1. Định nghĩa grid of hyperparameters:
   ```python
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, 30],
       'learning_rate': [0.01, 0.1, 0.3]
   }
   ```

2. Thử tất cả combinations (3 × 3 × 3 = 27 models)

3. Với mỗi combination:
   - Train model với parameters đó
   - Evaluate bằng cross-validation
   - Lưu performance score

4. Chọn combination có best score

**Ưu điểm**:
- Exhaustive search, đảm bảo tìm được best trong grid
- Systematic và reproducible

**Nhược điểm**:
- Rất chậm với grid lớn
- Chỉ thử các giá trị trong grid (có thể miss optimal values)

**Alternative**: Random Search - random sample combinations thay vì try tất cả

---

## 4. METRICS (CHỈ SỐ ĐÁNH GIÁ)

### 4.1. MAE (Mean Absolute Error)
**Công thức**: `MAE = (1/n) × Σ|yᵢ - ŷᵢ|`
**Ý nghĩa**: Sai số trung bình tuyệt đối
**Range**: [0, ∞), càng nhỏ càng tốt
**Ưu điểm**: Dễ hiểu, đơn vị giống target
**Nhược điểm**: Không phân biệt underestimate/overestimate

### 4.2. RMSE (Root Mean Squared Error)
**Công thức**: `RMSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]`
**Ý nghĩa**: Căn bậc hai của mean squared error
**Range**: [0, ∞), càng nhỏ càng tốt
**Ưu điểm**: Penalize lỗi lớn nhiều hơn MAE
**Nhược điểm**: Đơn vị là bình phương của target unit

### 4.3. R² Score (Coefficient of Determination)
**Công thức**: `R² = 1 - (SS_res / SS_tot)`
Trong đó:
- SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)

**Ý nghĩa**: Tỷ lệ variance được model giải thích
**Range**: (-∞, 1], càng gần 1 càng tốt
- R² = 1: Perfect predictions
- R² = 0: Model không tốt hơn baseline (mean)
- R² < 0: Model worse than baseline

**Ưu điểm**: 
- Normalized, dễ so sánh across datasets
- Intuitive interpretation

### 4.4. MAPE (Mean Absolute Percentage Error)
**Công thức**: `MAPE = (100/n) × Σ|（yᵢ - ŷᵢ）/yᵢ|`
**Ý nghĩa**: Sai số phần trăm trung bình
**Range**: [0, ∞), càng nhỏ càng tốt
**Ưu điểm**: Scale-independent, dễ hiểu với business
**Nhược điểm**: 
- Undefined khi yᵢ = 0
- Asymmetric (penalize underestimate nhiều hơn)

---

## 5. TÓM TẮT WORKFLOW

```
1. DATA LOADING
   ↓
2. EXPLORATORY DATA ANALYSIS
   ↓
3. DATA CLEANING
   - Remove unnecessary columns
   - Remove duplicates
   - Handle missing values
   - Remove outliers (IQR method)
   ↓
4. FEATURE ENGINEERING
   - Label encoding for categorical features
   - Feature scaling (if needed)
   ↓
5. TRAIN-TEST SPLIT (80/20)
   ↓
6. MODEL TRAINING
   - Linear Regression (baseline)
   - Random Forest (ensemble)
   - XGBoost (gradient boosting)
   - LightGBM (fast gradient boosting)
   ↓
7. MODEL EVALUATION
   - Calculate metrics (MAE, RMSE, R², MAPE)
   - Cross-validation
   - Compare models
   ↓
8. HYPERPARAMETER TUNING (optional)
   - Grid Search
   - Best parameters
   ↓
9. FINAL EVALUATION
   - Best model on test set
   - Feature importance analysis
   - Error analysis
   ↓
10. MODEL DEPLOYMENT
    - Save best model
    - Create prediction interface
```

---

## 6. BEST PRACTICES

### 6.1. Khi nào dùng model nào?

| Scenario | Recommended Model |
|----------|------------------|
| Cần interpretability cao | Linear Regression |
| Dữ liệu phức tạp, cần accuracy | Random Forest |
| Cần highest accuracy, có thời gian tune | XGBoost |
| Large dataset, cần speed | LightGBM |
| Small dataset (< 1K) | Random Forest or Linear |
| Imbalanced data | XGBoost or LightGBM |

### 6.2. Tips để tránh Overfitting

1. **Cross-validation**: Always use CV để detect overfitting
2. **Regularization**: Use L1/L2 trong XGBoost/LightGBM
3. **Early stopping**: Stop training khi validation error tăng
4. **Simpler models**: Giảm max_depth, tăng min_samples_split
5. **More data**: Collect thêm training data nếu có thể
6. **Feature selection**: Remove irrelevant features

### 6.3. Tips để improve Performance

1. **Feature Engineering**: Create new meaningful features
2. **Handle outliers**: Remove hoặc cap extreme values
3. **Feature scaling**: StandardScaler cho Linear models
4. **Ensemble methods**: Combine multiple models
5. **Hyperparameter tuning**: Systematic search for best params
6. **More data**: Always helps if available

---

## TÀI LIỆU THAM KHẢO

1. **Scikit-learn Documentation**: https://scikit-learn.org/
2. **XGBoost Documentation**: https://xgboost.readthedocs.io/
3. **LightGBM Documentation**: https://lightgbm.readthedocs.io/
4. **Random Forest Paper**: Breiman, L. (2001). Random Forests. Machine Learning.
5. **Gradient Boosting**: Friedman, J. H. (2001). Greedy Function Approximation.
