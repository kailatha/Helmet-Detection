# 🏍️ Deteksi Helm Pengendara Motor

Aplikasi web untuk mendeteksi penggunaan helm pada pengendara motor menggunakan model **YOLO11** dan **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLO-v11-00FFFF?logo=yolo&logoColor=white)

---

## 📋 Deskripsi

Proyek ini merupakan sistem deteksi objek berbasis **deep learning** yang mampu mengidentifikasi apakah pengendara motor menggunakan helm atau tidak. Model dilatih menggunakan arsitektur **YOLO11** pada dataset HELMET.

### Fitur Utama

- 📸 **Deteksi pada Gambar** — Unggah gambar dan lihat hasil deteksi secara langsung
- 🎥 **Deteksi pada Video** — Unggah video dan proses deteksi frame-by-frame
- ⚙️ **Pengaturan Fleksibel** — Atur confidence threshold dan ukuran input sesuai kebutuhan
- 📊 **Statistik Deteksi** — Tampilkan jumlah objek terdeteksi per kelas

---

## 📁 Struktur Proyek

```
helmet_detection/
├── app.py                 # Aplikasi utama Streamlit
├── models/
│   └── best.pt            # Model YOLO11 yang sudah dilatih
├── requirements.txt       # Daftar dependensi Python
├── .gitignore
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/kailatha/Helmet-Detection.git
cd Helmet-Detection
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Fungsi |
|-----------|--------|
| **Python** | Bahasa pemrograman utama |
| **YOLOv11 (Ultralytics)** | Model deteksi objek |
| **Streamlit** | Framework web untuk antarmuka pengguna |
| **OpenCV** | Pemrosesan gambar dan video |
| **Pillow** | Manipulasi gambar |
| **NumPy** | Komputasi numerik |

---

## 📖 Cara Penggunaan

1. Buka aplikasi di browser
2. Pilih tab **Gambar** atau **Video**
3. Unggah file yang ingin dideteksi
4. Atur **Confidence Threshold** dan **Ukuran Input** di sidebar sesuai kebutuhan
5. Lihat hasil deteksi dan statistik yang ditampilkan

### Pengaturan

- **Confidence Threshold** (0.10 – 0.90): Semakin tinggi, semakin ketat seleksi deteksi
- **Ukuran Input (imgsz)**: 640 / 800 / 960 piksel — ukuran 960 sesuai dengan ukuran saat training

---

## 🌐 Deploy di Streamlit Cloud

Aplikasi ini sudah siap di-deploy di [Streamlit Cloud](https://streamlit.io/cloud):

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Hubungkan repository dan pilih `app.py` sebagai main file
4. Klik **Deploy**

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan tugas Visi Komputer Kelompok 1.
