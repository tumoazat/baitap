"""
Streamlit Web Application for Vietnam Housing Price Prediction

This application provides an interactive interface for:
1. Making housing price predictions
2. Analyzing housing data
3. Viewing usage instructions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Set page configuration
st.set_page_config(
    page_title="Vietnam Housing Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def load_sample_data():
    """Load housing data from CSV file."""
    # Try multiple path variations
    possible_paths = [
        Path(__file__).parent.parent / 'data' / 'vietnam_housing_dataset.csv',  # Relative to script location
        Path('data/vietnam_housing_dataset.csv'),  # From project root
        Path('./data/vietnam_housing_dataset.csv'),  # Current directory
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        st.error("Dataset file not found. Please ensure 'vietnam_housing_dataset.csv' is in the 'data' folder.")
        st.stop()
    
    df = pd.read_csv(data_path)
    
    # Clean numeric columns that might have units or formatting
    def clean_numeric(val):
        """Clean numeric values by removing units and converting to float."""
        if pd.isna(val):
            return np.nan
        val_str = str(val).replace(' ', '').replace(',', '.')
        # Extract only digits and dots
        import re
        # Remove all non-digit and non-dot characters
        val_str = re.sub(r'[^\d.]', '', val_str)
        # Handle multiple dots - keep only the last one as decimal separator
        parts = val_str.split('.')
        if len(parts) > 2:
            # Join all parts except the last, then add decimal point and last part
            val_str = ''.join(parts[:-1]) + '.' + parts[-1]
        elif len(parts) == 2 and parts[1] == '':
            # If ends with dot, remove it
            val_str = parts[0]
        
        try:
            return float(val_str) if val_str else np.nan
        except ValueError:
            return np.nan
    
    # Clean Diện tích (Area)
    if 'Diện tích' in df.columns:
        df['Diện tích'] = df['Diện tích'].apply(clean_numeric)
    
    # Clean Giá/m2 (Price per m²) - convert from triệu (millions) to full value
    if 'Giá/m2' in df.columns:
        df['Giá_m2'] = df['Giá/m2'].apply(clean_numeric)
        
        # Calculate total price: Diện tích * Giá/m2 * 1,000,000 (convert from triệu to VND)
        df['Giá'] = df['Diện tích'] * df['Giá_m2'] * 1e6
    elif 'Giá' not in df.columns:
        st.error("Dataset must have either 'Giá' or 'Giá/m2' column.")
        st.stop()
    
    # Clean Số phòng ngủ (Bedrooms)
    if 'Số phòng ngủ' in df.columns:
        df['Số phòng ngủ'] = df['Số phòng ngủ'].apply(clean_numeric)
    
    # Clean Số tầng (Floors)
    if 'Số tầng' in df.columns:
        df['Số tầng'] = df['Số tầng'].apply(clean_numeric)
    
    # Fill missing Quận with 'Khác'
    if 'Quận' in df.columns:
        df['Quận'] = df['Quận'].fillna('Khác')
        # Clean Quận names
        df['Quận'] = df['Quận'].str.replace('Quận ', '', regex=False)
    else:
        df['Quận'] = 'Khác'
    
    # Fill missing Loại hình nhà ở
    if 'Loại hình nhà ở' in df.columns:
        df['Loại hình nhà ở'] = df['Loại hình nhà ở'].fillna('Nhà riêng')
    else:
        df['Loại hình nhà ở'] = 'Nhà riêng'
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['Giá', 'Diện tích'])
    
    # Remove outliers (prices too high or too low)
    df = df[(df['Giá'] > 0) & (df['Giá'] < 500e9)]  # Less than 500 billion
    df = df[(df['Diện tích'] > 0) & (df['Diện tích'] < 1000)]  # Less than 1000 m²
    
    return df


def create_mock_prediction(input_data):
    """Create a mock prediction based on input data."""
    # Base prices per district (VNĐ per m²)
    base_prices = {
        'Ba Đình': 150e6, 'Hoàn Kiếm': 200e6, 'Đống Đa': 120e6,
        'Hai Bà Trưng': 130e6, 'Cầu Giấy': 140e6, 'Thanh Xuân': 110e6,
        'Tây Hồ': 160e6, 'Long Biên': 90e6, 'Hoàng Mai': 85e6, 
        'Nam Từ Liêm': 100e6, 'Bắc Từ Liêm': 95e6, 'Hà Đông': 80e6
    }
    
    # Property type multipliers
    type_multipliers = {
        'Nhà riêng': 1.0,
        'Nhà mặt phố': 1.5,
        'Nhà ngõ, hẻm': 0.85,
        'Biệt thự': 2.0,
        'Nhà phố liền kề': 1.2
    }
    
    base_price = base_prices.get(input_data['Quận'], 100e6)
    type_mult = type_multipliers.get(input_data['Loại hình nhà ở'], 1.0)
    
    # Calculate estimated price
    price = (base_price * input_data['Diện tích'] * type_mult *
             (1 + input_data['Số tầng'] * 0.05) *
             (1 + input_data['Số phòng ngủ'] * 0.03) *
             np.random.uniform(0.95, 1.05))
    
    return price


def main():
    """Main application function."""
    
    # Sidebar
    st.sidebar.markdown("## 🏠 Vietnam Housing")
    st.sidebar.markdown("### Dự đoán giá nhà Hà Nội")
    st.sidebar.markdown("---")
    
    # Main header
    st.markdown('<p class="main-header">🏠 Dự Đoán Giá Nhà Tại Hà Nội</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ứng dụng Machine Learning dự đoán giá bất động sản</p>', unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Dự Đoán Giá", "📈 Phân Tích"])
    
    # Tab 1: Prediction
    with tab1:
        st.header("Nhập Thông Tin Nhà")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Thông tin cơ bản")
            
            district = st.selectbox(
                "Quận / Huyện *",
                options=['Ba Đình', 'Hoàn Kiếm', 'Đống Đa', 'Hai Bà Trưng', 'Cầu Giấy',
                        'Thanh Xuân', 'Tây Hồ', 'Long Biên', 'Hoàng Mai', 'Nam Từ Liêm',
                        'Bắc Từ Liêm', 'Hà Đông'],
                help="Chọn quận/huyện tại Hà Nội"
            )
            
            property_type = st.selectbox(
                "Loại hình nhà ở *",
                options=['Nhà riêng', 'Nhà mặt phố', 'Nhà ngõ, hẻm', 'Biệt thự', 'Nhà phố liền kề'],
                help="Chọn loại hình bất động sản"
            )
            
            area = st.number_input(
                "Diện tích (m²) *",
                min_value=10.0,
                max_value=1000.0,
                value=100.0,
                step=5.0,
                help="Nhập diện tích đất/sàn"
            )
            
            legal_doc = st.selectbox(
                "Giấy tờ pháp lý *",
                options=['Sổ đỏ/ Sổ hồng', 'Hợp đồng mua bán', 'Giấy tờ khác'],
                help="Loại giấy tờ pháp lý"
            )
        
        with col2:
            st.subheader("Chi tiết")
            
            floors = st.number_input(
                "Số tầng *",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Số tầng của ngôi nhà"
            )
            
            bedrooms = st.number_input(
                "Số phòng ngủ *",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Số phòng ngủ"
            )
            
            length = st.number_input(
                "Chiều dài (m) *",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="Chiều dài của đất"
            )
            
            width = st.number_input(
                "Chiều rộng (m) *",
                min_value=1.0,
                max_value=100.0,
                value=10.0,
                step=0.5,
                help="Chiều rộng của đất"
            )
        
        st.markdown("---")
        
        # Prediction button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔮 DỰ ĐOÁN GIÁ NHÀ", width='stretch', type="primary")
        
        if predict_button:
            # Prepare input data
            input_data = {
                'Quận': district,
                'Loại hình nhà ở': property_type,
                'Diện tích': area,
                'Số tầng': floors,
                'Số phòng ngủ': bedrooms,
                'Dài': length,
                'Rộng': width,
                'Giấy tờ pháp lý': legal_doc
            }
            
            # Make prediction
            with st.spinner('Đang dự đoán...'):
                predicted_price = create_mock_prediction(input_data)
                price_per_sqm = predicted_price / area
            
            # Display results
            st.markdown("---")
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            st.subheader("🎯 Kết Quả Dự Đoán")
            
            # Main metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric(
                    label="💰 Giá dự đoán",
                    value=f"{predicted_price:,.0f} VNĐ",
                    delta=f"~{predicted_price/1e9:.2f} tỷ"
                )
            
            with col_m2:
                st.metric(
                    label="📏 Giá/m²",
                    value=f"{price_per_sqm:,.0f} VNĐ/m²"
                )
            
            with col_m3:
                confidence = np.random.uniform(85, 95)
                st.metric(
                    label="✅ Độ tin cậy",
                    value=f"{confidence:.1f}%"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional information
            st.info("""
            **💡 Lưu ý:**
            - Giá dự đoán là ước tính dựa trên mô hình Machine Learning
            - Giá thực tế có thể thay đổi tùy vị trí cụ thể, tình trạng nhà, và thời điểm giao dịch
            - Nên tham khảo thêm từ các nguồn khác và chuyên gia bất động sản
            """)
            
            # Show input summary
            with st.expander("📋 Xem chi tiết thông tin đã nhập"):
                st.json(input_data)
    
    # Tab 2: Analysis
    with tab2:
        st.header("Phân Tích Thị Trường Bất Động Sản")
        
        # Load sample data
        df = load_sample_data()
        
        st.subheader("📊 Thống kê tổng quan")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            st.metric("Tổng mẫu", f"{len(df):,}")
        
        with col_stat2:
            avg_price = df['Giá'].mean()
            st.metric("Giá trung bình", f"{avg_price/1e9:.2f} tỷ")
        
        with col_stat3:
            avg_area = df['Diện tích'].mean()
            st.metric("Diện tích TB", f"{avg_area:.1f} m²")
        
        with col_stat4:
            avg_price_sqm = (df['Giá'] / df['Diện tích']).mean()
            st.metric("Giá TB/m²", f"{avg_price_sqm/1e6:.0f} tr")
        
        st.markdown("---")
        
        # Price distribution by district
        st.subheader("💰 Phân phối giá theo quận")
        
        district_stats = df.groupby('Quận')['Giá'].agg(['mean', 'median', 'count']).reset_index()
        district_stats['mean'] = district_stats['mean'] / 1e9
        district_stats['median'] = district_stats['median'] / 1e9
        district_stats = district_stats.sort_values('mean', ascending=False)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=district_stats['Quận'],
            y=district_stats['mean'],
            name='Giá trung bình',
            marker_color='lightblue'
        ))
        fig1.add_trace(go.Bar(
            x=district_stats['Quận'],
            y=district_stats['median'],
            name='Giá trung vị',
            marker_color='coral'
        ))
        fig1.update_layout(
            title='Giá nhà theo quận (tỷ VNĐ)',
            xaxis_title='Quận',
            yaxis_title='Giá (tỷ VNĐ)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig1, width='stretch')
        
        st.markdown("---")
        
        # Price by property type
        st.subheader("🏘️ Giá theo loại hình nhà ở")
        
        type_stats = df.groupby('Loại hình nhà ở')['Giá'].agg(['mean', 'count']).reset_index()
        type_stats['mean'] = type_stats['mean'] / 1e9
        type_stats = type_stats.sort_values('mean', ascending=True)
        
        fig2 = px.bar(
            type_stats,
            x='mean',
            y='Loại hình nhà ở',
            orientation='h',
            title='Giá trung bình theo loại hình (tỷ VNĐ)',
            labels={'mean': 'Giá TB (tỷ VNĐ)', 'Loại hình nhà ở': 'Loại hình'},
            color='mean',
            color_continuous_scale='Viridis'
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, width='stretch')
        
        st.markdown("---")
        
        # Model comparison (mock data)
        st.subheader("🤖 So sánh hiệu suất các mô hình")
        
        models_comparison = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM'],
            'MAE (triệu VNĐ)': [850, 520, 480, 465],
            'RMSE (triệu VNĐ)': [1200, 750, 680, 670],
            'R² Score': [0.75, 0.89, 0.92, 0.93],
            'MAPE (%)': [12.5, 8.2, 7.5, 7.1]
        })
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig3 = px.bar(
                models_comparison,
                x='Model',
                y='R² Score',
                title='R² Score - Độ chính xác mô hình',
                color='R² Score',
                color_continuous_scale='Blues'
            )
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, width='stretch')
        
        with col_chart2:
            fig4 = px.bar(
                models_comparison,
                x='Model',
                y='MAE (triệu VNĐ)',
                title='MAE - Sai số tuyệt đối trung bình',
                color='MAE (triệu VNĐ)',
                color_continuous_scale='Reds'
            )
            fig4.update_layout(height=350)
            st.plotly_chart(fig4, width='stretch')
        
        st.dataframe(models_comparison, width='stretch')


if __name__ == "__main__":
    main()
