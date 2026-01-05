# 📊 Sistem Analisis UMKM Indonesia

Platform Big Data untuk analisis tren harga dan penjualan UMKM Indonesia menggunakan **Google Colab** (GRATIS).

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| 📥 **Data Kaggle** | Integrasi dataset real Indonesia E-Commerce |
| 🔄 **Realtime Data** | Kurs USD/IDR dan indikator ekonomi |
| 🤖 **ML Predictions** | Prediksi penjualan 7 hari ke depan |
| 📦 **Product Clustering** | Segmentasi produk otomatis (K-Means) |
| 📊 **Visualizations** | Grafik interaktif dengan Plotly |
| 💾 **Export Rapih** | Excel (.xlsx) dan CSV dengan format Indonesia |

---

## 🚀 Quick Start

### Opsi 1: Google Colab (Rekomendasi)

1. **Buka Google Colab**: https://colab.research.google.com
2. **Upload notebook**: `notebooks/umkm_analysis_colab.ipynb`
3. **Jalankan**: `Runtime` → `Run all`
4. **Download hasil**: File Excel otomatis ter-download

### Opsi 2: Jalankan Lokal

```bash
# Clone repository
git clone https://github.com/your-repo/umkm-analytics.git
cd umkm-analytics

# Install dependencies
pip install -r requirements.txt

# Jalankan analisis
python notebooks/umkm_analysis_colab.py
```

---

## 📥 Menggunakan Dataset Kaggle (REAL Data)

Untuk menggunakan dataset nyata dari Kaggle:

### 1. Dapatkan API Key Kaggle
- Buka https://kaggle.com/settings
- Scroll ke bagian "API"
- Klik **"Create New Token"**
- File `kaggle.json` akan ter-download

### 2. Di Google Colab
- Jalankan notebook
- Upload `kaggle.json` saat diminta
- Dataset akan otomatis ter-download

### 3. Dataset yang Digunakan
- **Nama**: Indonesia E-Commerce Sales & Shipping 2023-2025
- **Link**: https://www.kaggle.com/datasets/bakitacos/indonesia-ecommerce-sales-shipping-20232025
- **Size**: 24 bulan data transaksi

> ⚠️ Jika Kaggle tidak tersedia, sistem otomatis menggunakan data simulasi

---

## 📁 Struktur Output

```
output/
├── umkm_analytics_results.xlsx  ⭐ REKOMENDASI
├── umkm_full_data.csv           (separator: ;)
├── umkm_product_segments.csv    (separator: ;)
└── umkm_sales_predictions.csv   (separator: ;)
```

### Format Excel (.xlsx)
| Sheet | Isi |
|-------|-----|
| Data Penjualan | 1000 baris pertama dataset |
| Segmentasi Produk | Hasil clustering produk |
| Prediksi 7 Hari | Forecast penjualan |
| Summary Kategori | Ringkasan per kategori |
| Summary Region | Ringkasan per wilayah |

### Format CSV
- **Separator**: Semicolon (`;`) - kompatibel Excel Indonesia
- **Encoding**: UTF-8 with BOM
- **Format angka**: `Rp 1.000.000` (titik sebagai pemisah ribuan)

---

## 🤖 Machine Learning Models

### 1. Sales Prediction
- **Algorithm**: Gradient Boosting Regressor
- **Features**:
  - Price, Discount, Rating
  - Day of week, Month, Weekend
  - Sales lag (1 day, 7 days)
  - Moving average (7 days)

### 2. Product Clustering
- **Algorithm**: K-Means (5 clusters)
- **Segments**:
  - ⭐ Best Sellers
  - 💎 Premium Products
  - 💰 Budget Products
  - 📦 Standard Products
  - 📉 Low Performers

---

## 📊 Contoh Output

### Prediksi 7 Hari
```
Tanggal;Hari;Prediksi_Sales;Prediksi_Revenue
2026-01-06;Tuesday;751;Rp 102.000.560
2026-01-10;Saturday;798;Rp 108.377.321
2026-01-11;Sunday;800;Rp 108.613.594
```

### Segmentasi Produk
```
product_id;price;sales_count;segment
PROD_0001;Rp 81.913;2770;Best Sellers
PROD_0002;Rp 95.666;2832;Budget Products
```

---

## 💡 Business Insights

1. **Weekend Effect**: Penjualan +10% di akhir pekan
2. **Holiday Season**: Desember-Januari +15%
3. **Optimal Discount**: 10-15% untuk konversi terbaik
4. **Focus Areas**: 
   - Best Sellers → Volume
   - Premium → Margin

---

## 📋 Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
plotly>=5.15.0
openpyxl>=3.1.0
requests>=2.31.0
kaggle>=1.5.0
```

---

## 🔧 Troubleshooting

### CSV tidak terpisah di Excel
1. Buka Excel
2. File → Open → pilih CSV
3. Data → Text to Columns
4. Pilih "Delimited" → Semicolon

### Kaggle download gagal
- Pastikan `kaggle.json` sudah di-upload
- Cek permission: `chmod 600 ~/.kaggle/kaggle.json`
- Atau gunakan data simulasi (otomatis)

### Matplotlib stuck
- Gunakan backend non-interactive: `matplotlib.use('Agg')`

---

## 📝 License

MIT License - Silakan gunakan dan modifikasi sesuai kebutuhan.

---

## 👥 Contributors

- **Project**: UMKM Analytics Platform
- **Purpose**: Analisis tren harga & penjualan untuk UMKM Indonesia
- **Platform**: Google Colab (Free)

---

**Last Updated**: January 2026