# Face Classification Deep Learning  
Sistem klasifikasi wajah berbasis deep learning menggunakan beberapa arsitektur modern seperti **ResNet100 ArcFace**, **Swin Transformer**.  Project ini mencakup pipeline pelatihan, evaluasi, visualisasi, serta inferensi menggunakan model terbaik.
---

## Team  

| NAMA | NIM |
|------|-----------|
|Zefanya Danovanta Tarigan|122140101|
|Kayla Chika Lathisya|122140009|
|Yohanna Anzelika Sitepu|122140010|
---

## 📖 Overview  

Project ini mengimplementasikan sistem face classification menggunakan pendekatan deep learning modern, terutama:

- **ArcFace + ResNet100** — Menghasilkan embedding wajah yang sangat diskriminatif.
- **Swin Transformer** — Vision Transformer dengan mekanisme shifted-window attention.

Fitur proyek ini mencakup:

- Evaluasi lengkap (Confusion Matrix, Accuracy/Loss Curve)
- Visualisasi hasil prediksi
- Training modular & dapat diperluas
- Model siap digunakan untuk inference

---
## 📁 Project Structure  
```
DeepLearningTubes/
│
├── 📂 Models/
│ └── swin_model.pth        # Model Swin Transformer 
│ └── Resnet100ArcFace.pth  # Model Resnet101 + ArcFace

├── 📂 Results/
│ ├── ArcResNet100/
│ │ ├── class_labels.json
│ │ ├── confusion_matrix.png
│ │ ├── loss_accuracy_plot.png
│ │ └── prediction_visuals_all_val.png
│ │
│ └── SwinTransformer/
│   ├── Confusion Matrix.png
│   ├── Grafik Accuracy.png
│   ├── Grafik Loss.png
│   └── Hasil Prediksi.png
│
├── 📄 swin_model.py        # Arsitektur Swin Transformer
├── 📄 PreprocessingImage.ipyb 
├── 📄 TrainResNet100.ipynb 
├── 📄 Dashboard.py 
├── 📄 requirements.txt
├── 📄 packages.txt
└── 📝 README.md
```
---

## 📊 Model Performance  

| Arsitektur | Validasi Akurasi | Pretrained |
|-----------|------------------|------------|
| **ResNet100 + ArcFace** | 80% | ImageNet-1K |
| **Swin Transformer** | 57% | ImageNet-1K |

---

### 🔧 Installation

```bash
# Clone repository
git clone https://github.com/danovantaa/FaceClassificationDeepLearning.git
cd DeepLearningTubes

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
``` 

---
## 📈 Menjalankan Inferensi (Streamlit Dashboard)

Untuk menguji model terbaik (**ResNet100 ArcFace**) secara interaktif, Anda dapat menggunakan **Streamlit Dashboard** yang telah disediakan.    

link Dashboard : [DASHBOARD](https://faceclassificationdeeplearning.streamlit.app/)
---

### 🔧 Menjalankan Dashboard Secara Lokal

Pastikan environment virtual Anda sudah aktif, kemudian jalankan perintah berikut:

```bash
streamlit run Dashboard.py
