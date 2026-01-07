"""
Module Xử lý Dữ liệu - Preprocessing
====================================


"""

import pandas as pd
import numpy as np
import re
from pathlib import Path


class HousingDataPreprocessor:
    """
    Class xử lý dữ liệu nhà ở
    
    Attributes:
        data_path: Đường dẫn đến file CSV
        df: DataFrame chứa dữ liệu
    """
    
    def __init__(self, data_path):
        """
        Khởi tạo processor
        
        Args:
            data_path: Đường dẫn file CSV
        """
        self.data_path = Path(data_path)
        self.df = None
        print(f"✅ Đã khởi tạo HousingDataPreprocessor với file: {data_path}")
    
    
    def load_data(self):
        """
        BƯỚC 1: Đọc dữ liệu từ file CSV
        
        Returns:
            DataFrame: Dữ liệu đã đọc
        """
        print("\n📂 Đang đọc dữ liệu...")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Đã đọc {len(self.df):,} dòng dữ liệu")
        print(f"📊 Số cột: {len(self.df.columns)}")
        
        return self.df
    
    
    def clean_numeric_column(self, column_name):
        """
        BƯỚC 2: Làm sạch một cột số (loại bỏ text, ký tự đặc biệt)
        
        Ví dụ: '50 m²' -> 50, '100,5' -> 100.5
        
        Args:
            column_name: Tên cột cần làm sạch
        """
        print(f"🧹 Đang làm sạch cột: {column_name}")
        
        def to_number(value):
            """Chuyển đổi giá trị thành số"""
            if pd.isna(value):
                return None
            
            # Chuyển về string và loại bỏ khoảng trắng
            text = str(value).replace(',', '.').replace(' ', '')
            
            # Tìm tất cả số trong text
            numbers = re.findall(r'\d+\.?\d*', text)
            
            if numbers:
                return float(numbers[0])
            return None
        
        # Áp dụng hàm chuyển đổi cho cả cột
        self.df[column_name] = self.df[column_name].apply(to_number)
        
        # Đếm số giá trị còn lại
        non_null = self.df[column_name].notna().sum()
        print(f"  ✓ Còn lại {non_null:,} giá trị hợp lệ")
    
    
    def calculate_price(self):
        """
        BƯỚC 3: Tính giá nhà từ Diện tích và Giá/m²
        
        Công thức: Giá = Diện tích × Giá/m² × 1,000,000
        """
        print("\n💰 Đang tính toán giá nhà...")
        
        if 'Diện tích' not in self.df.columns or 'Giá/m2' not in self.df.columns:
            print("⚠️  Thiếu cột Diện tích hoặc Giá/m2")
            return
        
        # Tính giá (đơn vị: VNĐ)
        self.df['Giá'] = self.df['Diện tích'] * self.df['Giá/m2'] * 1_000_000
        
        # Đếm số nhà đã tính được giá
        calculated = self.df['Giá'].notna().sum()
        print(f"✅ Đã tính giá cho {calculated:,} nhà")
    
    
    def remove_outliers(self, column, min_value, max_value):
        """
        BƯỚC 4: Loại bỏ outliers (giá trị bất thường)
        
        Outliers là những giá trị quá cao hoặc quá thấp so với thực tế.
        Ví dụ: Nhà 1m² hoặc 10,000m² là không hợp lý
        
        Args:
            column: Tên cột cần kiểm tra
            min_value: Giá trị tối thiểu chấp nhận được
            max_value: Giá trị tối đa chấp nhận được
        """
        print(f"\n🔍 Đang loại bỏ outliers cho cột: {column}")
        
        # Đếm số dòng ban đầu
        before = len(self.df)
        
        # Lọc dữ liệu: chỉ giữ giá trị trong khoảng [min, max]
        self.df = self.df[
            (self.df[column] >= min_value) & 
            (self.df[column] <= max_value)
        ]
        
        # Đếm số dòng sau khi lọc
        after = len(self.df)
        removed = before - after
        
        print(f"  ✓ Đã loại bỏ {removed:,} dòng bất thường")
        print(f"  ✓ Còn lại {after:,} dòng")
    
    
    def remove_missing_values(self, important_columns):
        """
        BƯỚC 5: Loại bỏ các dòng thiếu thông tin quan trọng
        
        Args:
            important_columns: Danh sách tên cột quan trọng (không được thiếu)
        """
        print(f"\n🔍 Đang loại bỏ dòng thiếu giá trị...")
        
        before = len(self.df)
        
        # Loại bỏ dòng có giá trị NaN ở các cột quan trọng
        self.df = self.df.dropna(subset=important_columns)
        
        after = len(self.df)
        removed = before - after
        
        print(f"  ✓ Đã loại bỏ {removed:,} dòng thiếu dữ liệu")
        print(f"  ✓ Còn lại {after:,} dòng hoàn chỉnh")
    
    
    def clean_district_names(self):
        """
        BƯỚC 6: Làm sạch tên quận (loại bỏ chữ "Quận")
        
        Ví dụ: "Quận Đống Đa" -> "Đống Đa"
        """
        print("\n🗺️  Đang làm sạch tên quận...")
        
        if 'Quận' not in self.df.columns:
            print("⚠️  Không tìm thấy cột Quận")
            return
        
        # Loại bỏ chữ "Quận " ở đầu
        self.df['Quận'] = self.df['Quận'].str.replace('Quận ', '', regex=False)
        
        # Đếm số quận khác nhau
        n_districts = self.df['Quận'].nunique()
        print(f"✅ Tìm thấy {n_districts} quận/huyện")
    
    
    def preprocess_all(self):
        """
        HÀM TỔNG HỢP: Chạy tất cả các bước xử lý
        
        Đây là hàm chính - gọi hàm này để xử lý toàn bộ dữ liệu
        
        Returns:
            DataFrame: Dữ liệu đã được xử lý hoàn chỉnh
        """
        print("\n" + "="*60)
        print("🚀 BẮT ĐẦU XỬ LÝ DỮ LIỆU")
        print("="*60)
        
        # Bước 1: Đọc dữ liệu
        self.load_data()
        
        # Bước 2: Làm sạch các cột số
        print("\n--- GIAI ĐOẠN 1: Làm sạch dữ liệu số ---")
        self.clean_numeric_column('Diện tích')
        self.clean_numeric_column('Giá/m2')
        self.clean_numeric_column('Số tầng')
        self.clean_numeric_column('Số phòng ngủ')
        
        # Bước 3: Tính giá nhà
        print("\n--- GIAI ĐOẠN 2: Tính toán ---")
        self.calculate_price()
        
        # Bước 4: Xóa các dòng thiếu thông tin quan trọng
        print("\n--- GIAI ĐOẠN 3: Loại bỏ dữ liệu thiếu ---")
        self.remove_missing_values(['Giá', 'Diện tích', 'Quận'])
        
        # Bước 5: Loại bỏ outliers
        print("\n--- GIAI ĐOẠN 4: Loại bỏ outliers ---")
        self.remove_outliers('Giá', min_value=500_000_000, max_value=100_000_000_000)
        self.remove_outliers('Diện tích', min_value=20, max_value=500)
        
        # Bước 6: Làm sạch tên quận
        print("\n--- GIAI ĐOẠN 5: Chuẩn hóa tên ---")
        self.clean_district_names()
        
        # Hoàn thành
        print("\n" + "="*60)
        print("✅ HOÀN THÀNH XỬ LÝ DỮ LIỆU")
        print(f"📊 Kết quả: {len(self.df):,} dòng × {len(self.df.columns)} cột")
        print("="*60 + "\n")
        
        return self.df
    
    
    def save_processed_data(self, output_path):
        """
        BƯỚC 7: Lưu dữ liệu đã xử lý ra file mới
        
        Args:
            output_path: Đường dẫn file CSV output
        """
        print(f"\n💾 Đang lưu dữ liệu vào: {output_path}")
        
        if self.df is None:
            print("⚠️  Chưa có dữ liệu để lưu!")
            return
        
        # Lưu ra file CSV
        self.df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # Tính kích thước file
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Đã lưu thành công! Kích thước: {file_size:.2f} MB")
    
    
    def get_summary_stats(self):
        """
        Hiển thị thống kê tổng quan về dữ liệu
        """
        if self.df is None:
            print("⚠️  Chưa có dữ liệu!")
            return
        
        print("\n" + "="*60)
        print("📈 THỐNG KÊ DỮ LIỆU")
        print("="*60)
        
        # Thống kê về giá
        print(f"\n💰 Giá nhà:")
        print(f"  • Trung bình: {self.df['Giá'].mean()/1e9:.2f} tỷ VNĐ")
        print(f"  • Thấp nhất: {self.df['Giá'].min()/1e9:.2f} tỷ VNĐ")
        print(f"  • Cao nhất: {self.df['Giá'].max()/1e9:.2f} tỷ VNĐ")
        
        # Thống kê về diện tích
        print(f"\n📐 Diện tích:")
        print(f"  • Trung bình: {self.df['Diện tích'].mean():.1f} m²")
        print(f"  • Nhỏ nhất: {self.df['Diện tích'].min():.1f} m²")
        print(f"  • Lớn nhất: {self.df['Diện tích'].max():.1f} m²")
        
        # Số lượng theo quận
        print(f"\n🗺️  Phân bố theo quận:")
        district_counts = self.df['Quận'].value_counts().head(5)
        for district, count in district_counts.items():
            print(f"  • {district}: {count:,} nhà")
        
        print("="*60 + "\n")


# ============================================================================
# PHẦN DEMO: Cách sử dụng module này
# ============================================================================

if __name__ == "__main__":
    """
    Demo cách sử dụng HousingDataPreprocessor
    
    Chạy file này để xem các bước xử lý dữ liệu
    """
    
    print("\n" + "🎓 "*20)
    print("DEMO: MODULE XỬ LÝ DỮ LIỆU NHÀ Ở")
    print("🎓 "*20 + "\n")
    
    # Bước 1: Khởi tạo processor
    data_path = Path(__file__).parent.parent / 'data' / 'vietnam_housing_dataset.csv'
    processor = HousingDataPreprocessor(data_path)
    
    # Bước 2: Xử lý toàn bộ dữ liệu
    df_processed = processor.preprocess_all()
    
    # Bước 3: Xem thống kê
    processor.get_summary_stats()
    
    # Bước 4: Lưu kết quả
    output_path = Path(__file__).parent.parent / 'data' / 'processed_housing_data.csv'
    processor.save_processed_data(output_path)
    
    print("\n✨ Hoàn thành demo!")
    print("💡 Bạn có thể import module này vào project khác để xử lý dữ liệu\n")
