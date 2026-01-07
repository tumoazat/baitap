"""
Ứng dụng Dự đoán Giá Nhà 
=============================================================

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ============================================================================
# PHẦN 1: CẤU HÌNH TRANG WEB
# ============================================================================

st.set_page_config(
    page_title="Dự đoán Giá Nhà Hà Nội",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Dự Đoán Giá Nhà Tại Hà Nội")
st.markdown("---")


# ============================================================================
# PHẦN 2: HÀM ĐỌC VÀ XỬ LÝ DỮ LIỆU
# ============================================================================

@st.cache_data  # Cache để tải nhanh hơn
def load_data():
    
    
    # Bước 1: Đọc file CSV
    data_path = Path(__file__).parent.parent / 'data' / 'vietnam_housing_dataset.csv'
    
    if not data_path.exists():
        st.error("❌ Không tìm thấy file dữ liệu!")
        st.stop()
    
    df = pd.read_csv(data_path)
    
    # Bước 2: Làm sạch dữ liệu
    # Hàm phụ để chuyển đổi text thành số
    def to_number(text):
        """Chuyển đổi text sang số (ví dụ: '50 m²' -> 50)"""
        if pd.isna(text):
            return None
        
        # Loại bỏ tất cả ký tự không phải số và dấu chấm
        import re
        text = str(text).replace(',', '.').replace(' ', '')
        numbers = re.findall(r'\d+\.?\d*', text)
        
        if numbers:
            return float(numbers[0])
        return None
    
    # Áp dụng hàm làm sạch cho các cột quan trọng
    df['Diện tích'] = df['Diện tích'].apply(to_number)
    df['Giá/m2'] = df['Giá/m2'].apply(to_number)
    df['Số tầng'] = df['Số tầng'].apply(to_number)
    
    # Bước 3: Tính giá nhà (Giá = Diện tích × Giá/m² × 1,000,000)
    df['Giá'] = df['Diện tích'] * df['Giá/m2'] * 1_000_000
    
    # Bước 4: Làm sạch tên quận (bỏ chữ "Quận")
    if 'Quận' in df.columns:
        df['Quận'] = df['Quận'].str.replace('Quận ', '', regex=False)
    
    # Bước 5: Chỉ giữ lại các dòng có đầy đủ thông tin quan trọng
    df = df.dropna(subset=['Giá', 'Diện tích', 'Quận'])
    
    # Bước 6: Loại bỏ outliers (giá trị bất thường)
    # Chỉ giữ nhà có giá từ 500 triệu đến 100 tỷ
    df = df[(df['Giá'] >= 500_000_000) & (df['Giá'] <= 100_000_000_000)]
    
    # Chỉ giữ nhà có diện tích từ 20m² đến 500m²
    df = df[(df['Diện tích'] >= 20) & (df['Diện tích'] <= 500)]
    
    return df


# ============================================================================
# PHẦN 3: HÀM DỰ ĐOÁN GIÁ ĐƠN GIẢN
# ============================================================================

def predict_price(district, area, floors, property_type, df):
   
    
    # Lọc các nhà tương tự theo quận
    similar_houses = df[df['Quận'] == district]
    
    # Nếu có thông tin loại hình, lọc thêm
    if property_type and 'Loại hình nhà ở' in df.columns:
        similar_houses = similar_houses[similar_houses['Loại hình nhà ở'] == property_type]
    
    # Nếu không tìm thấy nhà tương tự, dùng toàn bộ dataset
    if len(similar_houses) == 0:
        similar_houses = df
    
    # Tính giá trung bình mỗi m² của các nhà tương tự
    avg_price_per_sqm = (similar_houses['Giá'] / similar_houses['Diện tích']).mean()
    
    # Dự đoán giá = Diện tích × Giá TB/m²
    predicted_price = area * avg_price_per_sqm
    
    # Điều chỉnh theo số tầng (mỗi tầng thêm 5%)
    if floors:
        floor_factor = 1 + (floors - 1) * 0.05
        predicted_price *= floor_factor
    
    return predicted_price


# ============================================================================
# PHẦN 4: GIAO DIỆN CHÍNH
# ============================================================================

# Đọc dữ liệu
with st.spinner("⏳ Đang tải dữ liệu..."):
    df = load_data()

st.success(f"✅ Đã tải {len(df):,} bản ghi dữ liệu")

# Tạo 2 tabs: Dự đoán và Phân tích
tab1, tab2 = st.tabs(["🔮 Dự Đoán Giá", "📊 Phân Tích Dữ Liệu"])

# ----------------------------------------------------------------------------
# TAB 1: DỰ ĐOÁN GIÁ NHÀ
# ----------------------------------------------------------------------------
with tab1:
    st.header("Nhập thông tin nhà để dự đoán giá")
    
    # Tạo 2 cột để nhập liệu
    col1, col2 = st.columns(2)
    
    with col1:
        # Lấy danh sách quận từ dữ liệu
        districts = sorted(df['Quận'].unique())
        district = st.selectbox(
            "🏛️ Chọn Quận/Huyện",
            districts,
            help="Chọn khu vực bạn muốn mua nhà"
        )
        
        area = st.number_input(
            "📐 Diện tích (m²)",
            min_value=20,
            max_value=500,
            value=50,
            step=5,
            help="Nhập diện tích từ 20-500 m²"
        )
    
    with col2:
        floors = st.number_input(
            "🏢 Số tầng",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Nhà cao bao nhiêu tầng?"
        )
        
        # Lấy danh sách loại hình nhà từ dữ liệu
        property_types = ['Tất cả'] + sorted(df['Loại hình nhà ở'].unique().tolist())
        property_type = st.selectbox(
            "🏠 Loại hình nhà ở",
            property_types,
            help="Chọn loại nhà bạn muốn"
        )
    
    st.markdown("---")
    
    # Nút dự đoán
    if st.button("🔮 DỰ ĐOÁN GIÁ", type="primary", use_container_width=True):
        
        # Xử lý loại hình
        prop_type = None if property_type == 'Tất cả' else property_type
        
        # Gọi hàm dự đoán
        with st.spinner("Đang tính toán..."):
            predicted_price = predict_price(district, area, floors, prop_type, df)
        
        # Hiển thị kết quả
        st.markdown("### 💰 Kết quả dự đoán")
        
        # Tạo 3 cột để hiển thị thông tin
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Giá dự đoán",
                f"{predicted_price/1e9:.2f} tỷ",
                help="Đơn vị: tỷ VNĐ"
            )
        
        with col2:
            price_per_sqm = predicted_price / area
            st.metric(
                "Giá/m²",
                f"{price_per_sqm/1e6:.1f} triệu",
                help="Giá mỗi m²"
            )
        
        with col3:
            # So sánh với giá trung bình toàn thành phố
            city_avg = df['Giá'].mean()
            diff_percent = ((predicted_price - city_avg) / city_avg) * 100
            st.metric(
                "So với TB",
                f"{diff_percent:+.1f}%",
                help="So sánh với giá TB toàn Hà Nội"
            )
        
        # Thêm lời giải thích
        st.info(f"""
        📌 **Giải thích kết quả:**
        - Nhà ở {district}, diện tích {area}m², {floors} tầng
        - Giá dự đoán: **{predicted_price/1e9:.2f} tỷ VNĐ**
        - Giá này được tính dựa trên giá trung bình của {len(df[df['Quận']==district]):,} nhà tương tự trong khu vực
        """)

# ----------------------------------------------------------------------------
# TAB 2: PHÂN TÍCH DỮ LIỆU
# ----------------------------------------------------------------------------
with tab2:
    st.header("📊 Thống kê và Phân tích")
    
    # Thống kê tổng quan
    st.subheader("📈 Thống kê tổng quan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng số nhà", f"{len(df):,}")
    
    with col2:
        avg_price = df['Giá'].mean() / 1e9
        st.metric("Giá trung bình", f"{avg_price:.2f} tỷ")
    
    with col3:
        avg_area = df['Diện tích'].mean()
        st.metric("Diện tích TB", f"{avg_area:.0f} m²")
    
    with col4:
        avg_price_sqm = (df['Giá'] / df['Diện tích']).mean() / 1e6
        st.metric("Giá TB/m²", f"{avg_price_sqm:.0f} tr")
    
    st.markdown("---")
    
    # Biểu đồ 1: Phân bố giá theo quận
    st.subheader("💰 Giá trung bình theo Quận")
    
    # Tính giá trung bình mỗi quận
    price_by_district = df.groupby('Quận')['Giá'].mean().sort_values(ascending=False)
    price_by_district = price_by_district / 1e9  # Chuyển sang tỷ
    
    # Vẽ biểu đồ
    fig1 = px.bar(
        x=price_by_district.index,
        y=price_by_district.values,
        labels={'x': 'Quận', 'y': 'Giá trung bình (tỷ VNĐ)'},
        title='Giá nhà trung bình theo từng quận',
        color=price_by_district.values,
        color_continuous_scale='Blues'
    )
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    
    # Biểu đồ 2: Phân bố diện tích
    st.subheader("📐 Phân bố Diện tích")
    
    fig2 = px.histogram(
        df,
        x='Diện tích',
        nbins=50,
        title='Phân bố diện tích nhà',
        labels={'Diện tích': 'Diện tích (m²)', 'count': 'Số lượng nhà'},
        color_discrete_sequence=['#1f77b4']
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Biểu đồ 3: Mối quan hệ giá và diện tích
    st.subheader("📊 Mối quan hệ giữa Giá và Diện tích")
    
    # Lấy mẫu để vẽ nhanh hơn
    sample_df = df.sample(min(1000, len(df)))
    
    fig3 = px.scatter(
        sample_df,
        x='Diện tích',
        y=sample_df['Giá']/1e9,
        color='Quận',
        title='Giá nhà theo diện tích (màu khác nhau là quận khác nhau)',
        labels={'x': 'Diện tích (m²)', 'y': 'Giá (tỷ VNĐ)'},
        opacity=0.6
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)
    
    # Giải thích
    st.info("""
    💡 **Nhận xét:**
    - Giá nhà tăng theo diện tích (điều này rất hợp lý!)
    - Các quận trung tâm (màu khác) có giá cao hơn với cùng diện tích
    - Đây là lý do tại sao model ML có thể dự đoán giá dựa trên diện tích và vị trí
    """)


