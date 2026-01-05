# UMKM Analytics - Google Colab Notebook
# Menggunakan Dataset REAL dari Kaggle: Indonesia E-Commerce Sales
# Platform: Google Colab (GRATIS)

"""
## Cara Menggunakan:
1. Buka file ini di Google Colab: https://colab.research.google.com
2. Upload file atau buka dari GitHub
3. Jalankan semua cell dengan Runtime > Run all
4. Jika diminta Kaggle credentials, ikuti instruksi di cell pertama
"""

# ============================================
# SECTION 1: SETUP & DOWNLOAD KAGGLE DATASET
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score

# Visualization - use non-interactive backend for scripts
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False
    print("Plotly tidak tersedia, menggunakan matplotlib")

print("✅ Libraries loaded successfully!")
print(f"📅 Waktu analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================
# SECTION 2: DOWNLOAD DATASET DARI KAGGLE
# ============================================

def download_kaggle_dataset():
    """
    Download dataset dari Kaggle.
    Dataset: Indonesia E-Commerce Sales & Shipping 2023-2025
    """
    import subprocess
    import os
    
    print("\n" + "="*60)
    print("📥 DOWNLOADING KAGGLE DATASET")
    print("="*60)
    
    # Try to use kaggle API
    try:
        # Install kaggle if not available
        subprocess.run(['pip', 'install', 'kaggle', '-q'], capture_output=True)
        
        # Check if kaggle.json exists (for Colab, need to upload)
        kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
        
        if not os.path.exists(kaggle_path):
            print("""
⚠️ Kaggle API credentials tidak ditemukan!

Untuk menggunakan dataset Kaggle, ikuti langkah berikut:
1. Buka https://www.kaggle.com/settings
2. Scroll ke 'API' section
3. Klik 'Create New Token' - akan download kaggle.json
4. Upload kaggle.json ke Colab:
   - Klik folder icon di sidebar kiri
   - Upload kaggle.json
5. Jalankan cell ini lagi

Sementara, menggunakan dataset simulasi...
            """)
            return None
        
        # Set permissions
        os.chmod(kaggle_path, 0o600)
        
        # Download dataset
        print("📥 Downloading from Kaggle...")
        result = subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 
            'bakitacos/indonesia-ecommerce-sales-shipping-20232025',
            '--unzip', '-p', 'kaggle_data'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset downloaded successfully!")
            # Find the CSV file
            for f in os.listdir('kaggle_data'):
                if 'clean' in f.lower() and f.endswith('.csv'):
                    return f'kaggle_data/{f}'
            # If no clean file, use any CSV
            for f in os.listdir('kaggle_data'):
                if f.endswith('.csv'):
                    return f'kaggle_data/{f}'
        else:
            print(f"⚠️ Download failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"⚠️ Kaggle API error: {e}")
        return None

def load_kaggle_data(file_path):
    """
    Load dan preprocess dataset Kaggle
    """
    print(f"\n📂 Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"   Original columns: {list(df.columns)}")
    print(f"   Original shape: {df.shape}")
    
    # Standardize column names (lowercase, underscore)
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Map columns to our expected format
    # Adjust these mappings based on actual Kaggle dataset columns
    column_mapping = {
        'order_date': 'sale_date',
        'transaction_date': 'sale_date',
        'date': 'sale_date',
        'product_name': 'product_name',
        'product': 'product_name',
        'item_name': 'product_name',
        'category': 'category',
        'product_category': 'category',
        'price': 'price',
        'unit_price': 'price',
        'item_price': 'price',
        'quantity': 'sales_count',
        'qty': 'sales_count',
        'order_qty': 'sales_count',
        'discount': 'discount_percent',
        'discount_pct': 'discount_percent',
        'rating': 'rating',
        'review_rating': 'rating',
        'city': 'region',
        'province': 'region',
        'destination': 'region',
        'seller_id': 'seller_id',
        'merchant_id': 'seller_id',
        'shop_id': 'seller_id'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Create missing columns with defaults
    if 'product_id' not in df.columns:
        df['product_id'] = 'PROD_' + (df.index + 1).astype(str).str.zfill(4)
    
    if 'sale_date' not in df.columns:
        # Generate dates if not available
        df['sale_date'] = pd.date_range(end=datetime.now(), periods=len(df)).strftime('%Y-%m-%d')
    
    if 'price' not in df.columns:
        df['price'] = np.random.uniform(10000, 500000, len(df))
    
    if 'sales_count' not in df.columns:
        df['sales_count'] = np.random.randint(1, 50, len(df))
    
    if 'discount_percent' not in df.columns:
        df['discount_percent'] = np.random.choice([0, 5, 10, 15, 20], len(df), p=[0.7, 0.12, 0.09, 0.06, 0.03])
    
    if 'rating' not in df.columns:
        df['rating'] = np.round(np.random.uniform(3.0, 5.0, len(df)), 1)
    
    if 'category' not in df.columns:
        categories = ['Elektronik', 'Fashion', 'Makanan', 'Kesehatan', 'Rumah Tangga']
        df['category'] = np.random.choice(categories, len(df))
    
    if 'region' not in df.columns:
        regions = ['Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Bali', 'Sumatera']
        df['region'] = np.random.choice(regions, len(df))
    
    if 'seller_id' not in df.columns:
        df['seller_id'] = 'SELLER_' + np.random.randint(1, 100, len(df)).astype(str).str.zfill(3)
    
    # Calculate revenue
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(50000)
    df['sales_count'] = pd.to_numeric(df['sales_count'], errors='coerce').fillna(1).astype(int)
    df['discount_percent'] = pd.to_numeric(df['discount_percent'], errors='coerce').fillna(0)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(4.0)
    
    df['revenue'] = df['price'] * df['sales_count'] * (1 - df['discount_percent']/100)
    
    # Add date-based features
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df['day_of_week'] = df['sale_date'].dt.dayofweek
    df['month'] = df['sale_date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['sale_date'] = df['sale_date'].dt.strftime('%Y-%m-%d')
    
    # Select final columns
    final_columns = ['product_id', 'product_name', 'category', 'price', 'sales_count',
                     'discount_percent', 'rating', 'sale_date', 'region', 'seller_id',
                     'day_of_week', 'month', 'is_weekend', 'revenue']
    
    # Keep only columns that exist
    existing_cols = [c for c in final_columns if c in df.columns]
    df = df[existing_cols].copy()
    
    # Fill any remaining NaN
    df = df.fillna(0)
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"   Columns: {list(df.columns)}")
    
    return df

def generate_simulated_data(num_products=64, days_history=90):
    """
    Fallback: Generate simulated UMKM data if Kaggle not available
    """
    print("\n" + "="*60)
    print("📊 GENERATING SIMULATED DATA")
    print("="*60)
    
    categories = ['Makanan & Minuman', 'Pakaian', 'Kerajinan Tangan', 
                  'Elektronik', 'Pertanian', 'Jasa', 'Kosmetik', 'Furniture']
    regions = ['Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 
               'Bali', 'Sumatera Utara', 'Sulawesi Selatan', 'Kalimantan Timur']
    product_names = {
        'Makanan & Minuman': ['Kopi Arabika', 'Sambal Bu Rudy', 'Keripik Tempe', 'Dodol Garut'],
        'Pakaian': ['Batik Pekalongan', 'Tenun Lombok', 'Kebaya Modern', 'Sarung Samarinda'],
        'Kerajinan Tangan': ['Tas Rotan', 'Wayang Kulit', 'Ukiran Jepara', 'Anyaman Bambu'],
        'Elektronik': ['Charger Universal', 'Earphone Wireless', 'Power Bank', 'LED Strip'],
        'Pertanian': ['Beras Organik', 'Madu Hutan', 'Gula Aren', 'Kopi Robusta'],
        'Jasa': ['Laundry Kilat', 'Service AC', 'Cuci Motor', 'Fotocopy'],
        'Kosmetik': ['Lulur Tradisional', 'Masker Kopi', 'Lip Balm Alami', 'Sabun Herbal'],
        'Furniture': ['Kursi Rotan', 'Meja Jati', 'Rak Bambu', 'Lemari Minimalis']
    }
    
    data = []
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    
    product_id = 1
    for category in categories:
        for product_name in product_names[category][:num_products // len(categories)]:
            if category in ['Elektronik', 'Furniture']:
                base_price = np.random.uniform(100000, 500000)
            elif category in ['Jasa']:
                base_price = np.random.uniform(20000, 100000)
            else:
                base_price = np.random.uniform(25000, 150000)
            
            for day_offset in range(days_history):
                sale_date = datetime.now() - timedelta(days=days_history - day_offset)
                day_of_week = sale_date.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                month = sale_date.month
                
                seasonal = 1.15 if month in [12, 1] else (0.95 if month in [6, 7] else 1.0)
                weekend = 1.1 if is_weekend else 1.0
                
                price = round(base_price * seasonal * np.random.uniform(0.95, 1.05), -2)
                sales = max(1, int(np.random.poisson(20) * weekend * seasonal))
                discount = np.random.choice([0, 5, 10, 15, 20], p=[0.7, 0.12, 0.09, 0.06, 0.03])
                rating = round(min(5.0, max(1.0, 3.5 + np.random.normal(0, 0.5))), 1)
                
                data.append({
                    'product_id': f'PROD_{product_id:04d}',
                    'product_name': product_name,
                    'category': category,
                    'price': price,
                    'sales_count': sales,
                    'discount_percent': discount,
                    'rating': rating,
                    'sale_date': sale_date.strftime('%Y-%m-%d'),
                    'region': np.random.choice(regions),
                    'seller_id': f"SELLER_{product_id:03d}_{np.random.randint(1, 4)}",
                    'day_of_week': day_of_week,
                    'month': month,
                    'is_weekend': is_weekend,
                    'revenue': round(price * sales * (1 - discount/100), -2)
                })
            product_id += 1
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df):,} records for {df['product_id'].nunique()} products")
    return df

# ============================================
# MAIN: TRY KAGGLE, FALLBACK TO SIMULATED
# ============================================

# Try to download from Kaggle
kaggle_file = download_kaggle_dataset()

if kaggle_file:
    df = load_kaggle_data(kaggle_file)
    DATA_SOURCE = "Kaggle: Indonesia E-Commerce Sales 2023-2025"
else:
    df = generate_simulated_data(num_products=64, days_history=90)
    DATA_SOURCE = "Simulated UMKM Data"

print(f"\n📊 Data Source: {DATA_SOURCE}")
print(f"📋 Sample Data (5 baris pertama):")
print(df.head().to_string())

# ============================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "="*60)
print("📈 EXPLORATORY DATA ANALYSIS")
print("="*60)

# Basic Statistics
print("\n📊 Statistik Deskriptif:")
numeric_cols = ['price', 'sales_count', 'discount_percent', 'rating', 'revenue']
existing_numeric = [c for c in numeric_cols if c in df.columns]
print(df[existing_numeric].describe().round(2))

# Sales by Category
if 'category' in df.columns:
    print("\n📦 Penjualan per Kategori:")
    category_stats = df.groupby('category').agg({
        'sales_count': 'sum',
        'revenue': 'sum',
        'price': 'mean',
        'rating': 'mean'
    }).round(2)
    category_stats.columns = ['Total Sales', 'Total Revenue', 'Avg Price', 'Avg Rating']
    category_stats = category_stats.sort_values('Total Revenue', ascending=False)
    print(category_stats)

# Sales by Region
if 'region' in df.columns:
    print("\n🗺️ Penjualan per Region:")
    region_stats = df.groupby('region').agg({
        'sales_count': 'sum',
        'revenue': 'sum'
    }).sort_values('revenue', ascending=False)
    region_stats.columns = ['Total Sales', 'Total Revenue']
    print(region_stats)

# ============================================
# SECTION 4: VISUALIZATIONS
# ============================================

print("\n" + "="*60)
print("📊 VISUALIZATIONS")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sales Trend
if 'sale_date' in df.columns:
    daily_sales = df.groupby('sale_date')['sales_count'].sum().reset_index()
    daily_sales['sale_date'] = pd.to_datetime(daily_sales['sale_date'])
    axes[0, 0].plot(daily_sales['sale_date'], daily_sales['sales_count'], color='#2596be', linewidth=2)
    axes[0, 0].fill_between(daily_sales['sale_date'], daily_sales['sales_count'], alpha=0.3, color='#2596be')
axes[0, 0].set_title('📈 Trend Penjualan Harian', fontsize=12, fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. Revenue by Category
if 'category' in df.columns:
    cat_revenue = df.groupby('category')['revenue'].sum().sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cat_revenue)))
    axes[0, 1].barh(cat_revenue.index, cat_revenue.values / 1e6, color=colors)
axes[0, 1].set_title('💰 Revenue per Kategori (Juta Rp)', fontsize=12, fontweight='bold')

# 3. Price Distribution
axes[1, 0].hist(df['price'] / 1000, bins=30, color='#ff6b6b', edgecolor='white', alpha=0.7)
axes[1, 0].set_title('📊 Distribusi Harga (Ribu Rp)', fontsize=12, fontweight='bold')

# 4. Rating Distribution
axes[1, 1].hist(df['rating'], bins=20, color='#4ecdc4', edgecolor='white', alpha=0.7)
axes[1, 1].set_title('⭐ Distribusi Rating', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('umkm_analysis_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Visualisasi disimpan ke 'umkm_analysis_charts.png'")

# ============================================
# SECTION 5: MACHINE LEARNING MODELS
# ============================================

print("\n" + "="*60)
print("🤖 MACHINE LEARNING MODELS")
print("="*60)

# Prepare ML Features
print("\n🔧 Preparing features for ML...")

ml_data = df.copy()

# Create lag features if we have time series
if 'product_id' in ml_data.columns and 'sale_date' in ml_data.columns:
    ml_data = ml_data.sort_values(['product_id', 'sale_date'])
    ml_data['sales_lag_1'] = ml_data.groupby('product_id')['sales_count'].shift(1)
    ml_data['sales_lag_7'] = ml_data.groupby('product_id')['sales_count'].shift(7)
    ml_data['sales_ma_7'] = ml_data.groupby('product_id')['sales_count'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
else:
    ml_data['sales_lag_1'] = ml_data['sales_count'].shift(1)
    ml_data['sales_lag_7'] = ml_data['sales_count'].shift(7)
    ml_data['sales_ma_7'] = ml_data['sales_count'].rolling(7, min_periods=1).mean()

# Category encoding
if 'category' in ml_data.columns:
    ml_data['category_encoded'] = LabelEncoder().fit_transform(ml_data['category'].astype(str))
else:
    ml_data['category_encoded'] = 0

# Fill NaN and prepare features
ml_data = ml_data.dropna()

print(f"   📊 Total samples for ML: {len(ml_data):,}")

# Features for prediction
feature_cols = ['price', 'discount_percent', 'rating', 'day_of_week', 'month', 
                'is_weekend', 'sales_lag_1', 'sales_lag_7', 'sales_ma_7', 'category_encoded']
existing_features = [c for c in feature_cols if c in ml_data.columns]

if len(existing_features) >= 5 and len(ml_data) > 100:
    X = ml_data[existing_features]
    y = ml_data['sales_count']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Testing samples: {len(X_test):,}")
    
    # Linear Regression
    print("\n   🔹 Linear Regression:")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    print(f"      R² Score: {lr_r2:.4f}")
    
    # Gradient Boosting
    print("\n   🔹 Gradient Boosting Regressor:")
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_r2 = r2_score(y_test, gb_pred)
    print(f"      R² Score: {gb_r2:.4f}")
    
    # Feature Importance
    print("\n   📊 Feature Importance (Top 5):")
    importance_df = pd.DataFrame({
        'feature': existing_features,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head().to_string(index=False))
else:
    print("⚠️ Not enough data/features for ML training")
    gb_model = None
    gb_r2 = 0

# Product Clustering
print("\n" + "-"*40)
print("📊 Product Clustering")
print("-"*40)

product_features = df.groupby('product_id').agg({
    'price': 'mean',
    'sales_count': 'sum',
    'rating': 'mean',
    'discount_percent': 'mean',
    'revenue': 'sum'
}).reset_index()

if len(product_features) >= 5:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(product_features[['price', 'sales_count', 'rating', 'discount_percent', 'revenue']])
    
    n_clusters = min(5, len(product_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    product_features['cluster'] = kmeans.fit_predict(features_scaled)
    
    silhouette = silhouette_score(features_scaled, product_features['cluster'])
    print(f"   Silhouette Score: {silhouette:.4f}")
    
    # Add segment names
    segment_names = {0: 'Standard', 1: 'Premium', 2: 'Budget', 3: 'Best Sellers', 4: 'Low Performers'}
    product_features['segment'] = product_features['cluster'].map(lambda x: segment_names.get(x, 'Other'))
else:
    product_features['cluster'] = 0
    product_features['segment'] = 'All Products'

# ============================================
# SECTION 6: PREDICTIONS
# ============================================

print("\n" + "="*60)
print("🔮 PREDICTIONS & INSIGHTS")
print("="*60)

if gb_model and 'product_id' in ml_data.columns:
    print("\n📅 Sales Prediction untuk 7 Hari ke Depan:")
    latest_data = ml_data.groupby('product_id').last().reset_index()
    
    future_predictions = []
    for day in range(1, 8):
        future_date = datetime.now() + timedelta(days=day)
        pred_features = latest_data[existing_features].copy()
        pred_features['day_of_week'] = future_date.weekday()
        pred_features['month'] = future_date.month
        pred_features['is_weekend'] = 1 if future_date.weekday() >= 5 else 0
        
        predictions = gb_model.predict(pred_features)
        future_predictions.append({
            'Tanggal': future_date.strftime('%Y-%m-%d'),
            'Hari': future_date.strftime('%A'),
            'Prediksi_Sales': int(predictions.sum()),
            'Prediksi_Revenue': int(predictions.sum() * latest_data['price'].mean() * 0.9)
        })
    
    predictions_df = pd.DataFrame(future_predictions)
    print(predictions_df.to_string(index=False))
else:
    # Simple prediction based on averages
    avg_daily_sales = df['sales_count'].sum() / df['sale_date'].nunique() if 'sale_date' in df.columns else df['sales_count'].mean()
    avg_price = df['price'].mean()
    
    future_predictions = []
    for day in range(1, 8):
        future_date = datetime.now() + timedelta(days=day)
        weekend_mult = 1.1 if future_date.weekday() >= 5 else 1.0
        predicted_sales = int(avg_daily_sales * weekend_mult)
        future_predictions.append({
            'Tanggal': future_date.strftime('%Y-%m-%d'),
            'Hari': future_date.strftime('%A'),
            'Prediksi_Sales': predicted_sales,
            'Prediksi_Revenue': int(predicted_sales * avg_price * 0.9)
        })
    predictions_df = pd.DataFrame(future_predictions)
    print(predictions_df.to_string(index=False))

# ============================================
# SECTION 7: EXPORT RESULTS (FORMAT RAPIH)
# ============================================

print("\n" + "="*60)
print("💾 EXPORT RESULTS")
print("="*60)

# 1. Format data untuk export
df_export = df.copy()
df_export['price_formatted'] = df_export['price'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
df_export['revenue_formatted'] = df_export['revenue'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))

# 2. Format product segments
product_export = product_features.copy()
product_export['price_formatted'] = product_export['price'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
product_export['revenue_formatted'] = product_export['revenue'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))
product_export['sales_formatted'] = product_export['sales_count'].apply(lambda x: f"{x:,.0f}".replace(",", "."))

# 3. Format predictions
pred_export = predictions_df.copy()
pred_export['Prediksi_Sales_Formatted'] = pred_export['Prediksi_Sales'].apply(lambda x: f"{x:,}".replace(",", "."))
pred_export['Prediksi_Revenue_Formatted'] = pred_export['Prediksi_Revenue'].apply(lambda x: f"Rp {x:,.0f}".replace(",", "."))

# Export CSV dengan semicolon separator
import os
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

df_export.to_csv(f'{output_dir}/umkm_full_data.csv', index=False, sep=';', encoding='utf-8-sig')
product_export.to_csv(f'{output_dir}/umkm_product_segments.csv', index=False, sep=';', encoding='utf-8-sig')
pred_export.to_csv(f'{output_dir}/umkm_sales_predictions.csv', index=False, sep=';', encoding='utf-8-sig')

# Export Excel
try:
    import subprocess
    subprocess.run(['pip', 'install', 'openpyxl', '-q'], capture_output=True)
    
    with pd.ExcelWriter(f'{output_dir}/umkm_analytics_results.xlsx', engine='openpyxl') as writer:
        df.head(1000).to_excel(writer, sheet_name='Data Penjualan', index=False)
        product_features.to_excel(writer, sheet_name='Segmentasi Produk', index=False)
        predictions_df.to_excel(writer, sheet_name='Prediksi 7 Hari', index=False)
        if 'category' in df.columns:
            category_stats.reset_index().to_excel(writer, sheet_name='Summary Kategori', index=False)
        if 'region' in df.columns:
            region_stats.reset_index().to_excel(writer, sheet_name='Summary Region', index=False)
    
    print(f"✅ Excel file created: {output_dir}/umkm_analytics_results.xlsx")
except Exception as e:
    print(f"⚠️ Excel export failed: {e}")

print(f"\n✅ Data exported to '{output_dir}/' folder:")
print(f"   📁 {output_dir}/umkm_full_data.csv (separator: ;)")
print(f"   📁 {output_dir}/umkm_product_segments.csv")
print(f"   📁 {output_dir}/umkm_sales_predictions.csv")
print(f"   📁 {output_dir}/umkm_analytics_results.xlsx ⭐")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("🎉 ANALISIS SELESAI!")
print("="*60)
print(f"""
📊 Summary:
   - Data Source: {DATA_SOURCE}
   - Total Records: {len(df):,}
   - Products: {df['product_id'].nunique() if 'product_id' in df.columns else 'N/A'}
   - Categories: {df['category'].nunique() if 'category' in df.columns else 'N/A'}
   - Best Model R²: {gb_r2:.4f}
   
🔗 Files:
   - umkm_analytics_results.xlsx (REKOMENDASI)
   - umkm_full_data.csv
   - umkm_product_segments.csv
   - umkm_sales_predictions.csv
""")
