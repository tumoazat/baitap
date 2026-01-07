"""
Module Huấn luyện Model Machine Learning 
================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Import các thư viện ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


class SimpleHousingModel:
    """
    Class huấn luyện model dự đoán giá nhà - Phiên bản đơn giản
    
    Attributes:
        df: DataFrame chứa dữ liệu
        X_train, X_test: Dữ liệu features để train và test
        y_train, y_test: Giá trị target (giá nhà) để train và test
        models: Dictionary chứa các model đã train
    """
    
    def __init__(self):
        """Khởi tạo class"""
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        
        print("✅ Đã khởi tạo SimpleHousingModel")
    
    
    def load_data(self, data_path):
        """
        BƯỚC 1: Đọc dữ liệu đã được xử lý
        
        Args:
            data_path: Đường dẫn đến file CSV
        """
        print(f"\n📂 Đang đọc dữ liệu từ: {data_path}")
        
        self.df = pd.read_csv(data_path)
        print(f"✅ Đã đọc {len(self.df):,} dòng dữ liệu")
        
        return self.df
    
    
    def prepare_features(self):
        """
        BƯỚC 2: Chuẩn bị features (đặc trưng) cho model
        
        Features là những thông tin dùng để dự đoán giá nhà.
        Trong ví dụ này chúng ta chỉ dùng 2 features đơn giản:
        - Diện tích (m²)
        - Số tầng
        
        Target (mục tiêu dự đoán): Giá nhà
        """
        print("\n🔧 Đang chuẩn bị features...")
        
        # Chọn các cột features (X)
        feature_columns = ['Diện tích', 'Số tầng']
        
        # Kiểm tra xem có đủ cột không
        missing_cols = [col for col in feature_columns if col not in self.df.columns]
        if missing_cols:
            print(f"⚠️  Thiếu cột: {missing_cols}")
            return False
        
        # Lấy features (X) và target (y)
        X = self.df[feature_columns].copy()
        y = self.df['Giá'].copy()  # Giá nhà là target
        
        # Xử lý giá trị NaN (thiếu)
        # Điền giá trị trung bình vào chỗ thiếu
        for col in feature_columns:
            if X[col].isna().any():
                mean_value = X[col].mean()
                X[col].fillna(mean_value, inplace=True)
                print(f"  ✓ Đã điền giá trị thiếu cho {col}")
        
        # Loại bỏ các dòng có giá = NaN
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Đã chuẩn bị {len(X):,} samples với {len(feature_columns)} features")
        print(f"   Features: {', '.join(feature_columns)}")
        
        return X, y
    
    
    def split_data(self, X, y, test_size=0.2):
        """
        BƯỚC 3: Chia dữ liệu thành tập train và test
        
        - Train set (80%): Dùng để huấn luyện model
        - Test set (20%): Dùng để đánh giá model
        
        Tại sao phải chia?
        -> Để kiểm tra xem model có hoạt động tốt với dữ liệu MỚI không
        
        Args:
            X: Features
            y: Target (giá nhà)
            test_size: Tỷ lệ dữ liệu dùng để test (mặc định 20%)
        """
        print(f"\n✂️  Đang chia dữ liệu ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=42  # Để kết quả có thể lặp lại
        )
        
        print(f"✅ Train set: {len(self.X_train):,} samples")
        print(f"✅ Test set: {len(self.X_test):,} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    
    def train_linear_regression(self):
        """
        BƯỚC 4A: Huấn luyện model Linear Regression
        
        Linear Regression là thuật toán đơn giản nhất trong ML.
        Nó tìm một đường thẳng (hoặc mặt phẳng) để fit với dữ liệu.
        
        Công thức: y = a*x1 + b*x2 + c
        Với y = giá nhà, x1 = diện tích, x2 = số tầng
        """
        print("\n🤖 Đang huấn luyện Linear Regression...")
        
        # Tạo model
        model = LinearRegression()
        
        # Huấn luyện model (fit)
        model.fit(self.X_train, self.y_train)
        
        # Lưu model
        self.models['Linear Regression'] = model
        
        # In ra các hệ số (coefficients)
        print("✅ Hoàn thành huấn luyện!")
        print(f"   Hệ số (coefficients):")
        for i, col in enumerate(self.X_train.columns):
            print(f"     • {col}: {model.coef_[i]:.2f}")
        print(f"   Hệ số tự do (intercept): {model.intercept_:.2f}")
        
        return model
    
    
    def train_random_forest(self, n_trees=100):
        """
        BƯỚC 4B: Huấn luyện model Random Forest
        
        Random Forest mạnh hơn Linear Regression.
        Nó tạo ra nhiều "cây quyết định" và kết hợp kết quả của chúng.
        
        Args:
            n_trees: Số lượng cây (mặc định 100)
        """
        print(f"\n🌲 Đang huấn luyện Random Forest ({n_trees} cây)...")
        
        # Tạo model
        model = RandomForestRegressor(
            n_estimators=n_trees,  # Số cây
            max_depth=10,          # Độ sâu tối đa của mỗi cây
            random_state=42,       # Để kết quả lặp lại được
            n_jobs=-1             # Dùng hết CPU
        )
        
        # Huấn luyện model
        model.fit(self.X_train, self.y_train)
        
        # Lưu model
        self.models['Random Forest'] = model
        
        print("✅ Hoàn thành huấn luyện!")
        
        # In ra feature importance (features nào quan trọng nhất)
        print("   Feature Importance:")
        for i, col in enumerate(self.X_train.columns):
            importance = model.feature_importances_[i] * 100
            print(f"     • {col}: {importance:.1f}%")
        
        return model
    
    
    def evaluate_model(self, model_name):
        """
        BƯỚC 5: Đánh giá model
        
        Dùng 3 chỉ số:
        - MAE (Mean Absolute Error): Sai số trung bình (càng nhỏ càng tốt)
        - RMSE (Root Mean Squared Error): Phạt nặng sai số lớn (càng nhỏ càng tốt)
        - R² Score: Model giải thích được bao nhiêu % biến động giá (0-1, càng gần 1 càng tốt)
        
        Args:
            model_name: Tên model cần đánh giá
        """
        if model_name not in self.models:
            print(f"⚠️  Không tìm thấy model: {model_name}")
            return None
        
        print(f"\n📊 Đang đánh giá model: {model_name}")
        
        model = self.models[model_name]
        
        # Dự đoán trên tập test
        y_pred = model.predict(self.X_test)
        
        # Tính các metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        
        # Tính MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((self.y_test - y_pred) / self.y_test)) * 100
        
        # In kết quả
        print(f"\n{'='*50}")
        print(f"KẾT QUẢ ĐÁNH GIÁ: {model_name}")
        print(f"{'='*50}")
        print(f"📏 MAE:   {mae/1e9:.3f} tỷ VNĐ")
        print(f"   (Sai số trung bình mỗi dự đoán)")
        print(f"\n📏 RMSE:  {rmse/1e9:.3f} tỷ VNĐ")
        print(f"   (Sai số với trọng số cao hơn)")
        print(f"\n📈 R²:    {r2:.4f} ({r2*100:.2f}%)")
        print(f"   (Model giải thích được {r2*100:.2f}% biến động giá)")
        print(f"\n📊 MAPE:  {mape:.2f}%")
        print(f"   (Sai số trung bình theo phần trăm)")
        print(f"{'='*50}\n")
        
        # Lưu kết quả
        results = {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        return results
    
    
    def compare_models(self):
        """
        BƯỚC 6: So sánh các models
        
        Tạo bảng so sánh để xem model nào tốt nhất
        """
        if len(self.models) == 0:
            print("⚠️  Chưa có model nào được huấn luyện!")
            return
        
        print("\n" + "="*60)
        print("🏆 SO SÁNH CÁC MODELS")
        print("="*60 + "\n")
        
        results_list = []
        
        # Đánh giá tất cả models
        for model_name in self.models.keys():
            results = self.evaluate_model(model_name)
            if results:
                results_list.append(results)
        
        # Tạo DataFrame để so sánh
        df_results = pd.DataFrame(results_list)
        df_results = df_results.sort_values('r2', ascending=False)
        
        print("\n📊 Bảng so sánh:")
        print(df_results.to_string(index=False))
        
        # Chọn model tốt nhất
        best_model = df_results.iloc[0]['model']
        best_r2 = df_results.iloc[0]['r2']
        
        print(f"\n🏆 Model tốt nhất: {best_model} (R² = {best_r2:.4f})")
        
        return df_results
    
    
    def save_model(self, model_name, output_path):
        """
        BƯỚC 7: Lưu model để dùng sau
        
        Args:
            model_name: Tên model cần lưu
            output_path: Đường dẫn file output
        """
        if model_name not in self.models:
            print(f"⚠️  Không tìm thấy model: {model_name}")
            return
        
        print(f"\n💾 Đang lưu model: {model_name}")
        
        model = self.models[model_name]
        
        # Lưu model bằng joblib
        joblib.dump(model, output_path)
        
        # Tính kích thước file
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        
        print(f"✅ Đã lưu model tại: {output_path}")
        print(f"   Kích thước: {file_size:.2f} MB")
    
    
    def predict(self, model_name, dien_tich, so_tang):
        """
        Dự đoán giá nhà mới
        
        Args:
            model_name: Tên model dùng để dự đoán
            dien_tich: Diện tích nhà (m²)
            so_tang: Số tầng
            
        Returns:
            float: Giá dự đoán (VNĐ)
        """
        if model_name not in self.models:
            print(f"⚠️  Không tìm thấy model: {model_name}")
            return None
        
        model = self.models[model_name]
        
        # Chuẩn bị input
        X_new = pd.DataFrame({
            'Diện tích': [dien_tich],
            'Số tầng': [so_tang]
        })
        
        # Dự đoán
        predicted_price = model.predict(X_new)[0]
        
        return predicted_price


# ============================================================================
# PHẦN DEMO: Cách sử dụng module này
# ============================================================================

if __name__ == "__main__":
    """
    Demo đầy đủ: Từ đọc dữ liệu -> Huấn luyện -> Đánh giá -> Lưu model
    """
    
    print("\n" + "🎓 "*20)
    print("DEMO: HUẤN LUYỆN MODEL DỰ ĐOÁN GIÁ NHÀ")
    print("🎓 "*20 + "\n")
    
    # Khởi tạo
    model_trainer = SimpleHousingModel()
    
    # Bước 1: Đọc dữ liệu
    data_path = Path(__file__).parent.parent / 'data' / 'processed_housing_data.csv'
    
    if not data_path.exists():
        print(f"⚠️  File không tồn tại: {data_path}")
        print("💡 Hãy chạy preprocessing_simple.py trước!")
    else:
        model_trainer.load_data(data_path)
        
        # Bước 2: Chuẩn bị features
        X, y = model_trainer.prepare_features()
        
        # Bước 3: Chia train/test
        model_trainer.split_data(X, y, test_size=0.2)
        
        # Bước 4: Huấn luyện models
        model_trainer.train_linear_regression()
        model_trainer.train_random_forest(n_trees=100)
        
        # Bước 5: So sánh models
        model_trainer.compare_models()
        
        # Bước 6: Lưu model tốt nhất
        output_dir = Path(__file__).parent.parent / 'models'
        output_dir.mkdir(exist_ok=True)
        
        model_trainer.save_model(
            'Random Forest',
            output_dir / 'simple_housing_model.pkl'
        )
        
        # Bước 7: Demo dự đoán
        print("\n" + "="*60)
        print("🔮 DEMO DỰ ĐOÁN")
        print("="*60)
        
        # Dự đoán giá nhà 50m², 3 tầng
        predicted_price = model_trainer.predict('Random Forest', dien_tich=50, so_tang=3)
        print(f"\n🏠 Nhà 50m², 3 tầng")
        print(f"💰 Giá dự đoán: {predicted_price/1e9:.2f} tỷ VNĐ")
        
        # Dự đoán giá nhà 100m², 5 tầng
        predicted_price = model_trainer.predict('Random Forest', dien_tich=100, so_tang=5)
        print(f"\n🏠 Nhà 100m², 5 tầng")
        print(f"💰 Giá dự đoán: {predicted_price/1e9:.2f} tỷ VNĐ")
        
        print("\n✨ Hoàn thành demo!")
        print("💡 Bạn có thể dùng model đã lưu để dự đoán sau này\n")
