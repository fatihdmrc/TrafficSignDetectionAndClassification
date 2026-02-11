# Trafik Tabelası Tespiti ve Sınıflandırma Sistemi Model Eğitim Süreci

Bu proje, trafik sahnelerinde bulunan tabelaların:

1. 📍 Yer Tespiti (Localization)
2. 🧠 Sınıflandırılması (Classification)

işlemlerini gerçekleştiren uçtan uca bir bilgisayarlı görü sistemidir.

---

# 🏗️ Sistem Genel Mimarisi

```
Görsel / Video
        │
        ▼
YOLOv5 (Tespit - GTSDB)
        │
        ▼
Bounding Box Crop
        │
        ▼
CNN (Sınıflandırma - GTSRB)
```

---

# 📍 1️⃣ Yer Tespiti Modeli (Localization)

## Kullanılan Veri Seti

German Traffic Sign Detection Benchmark (GTSDB)

- 600 adet trafik sahnesi görseli
- Etiket dosyaları: `gt.txt`, `ex.txt`
- Format:

```
<filename>;X1;Y1;X2;Y2;class_id
```

---

## 🎯 Model Amacı

Sınıf ayrımı yapmadan yalnızca trafik tabelalarının konumlarını tespit eden, hafif ve gerçek zamanlı çalışabilen bir model geliştirmek.

---

## 🧩 Veri Hazırlığı

- `.ppm → .jpg` dönüşümü yapıldı
- `gt.txt` + `ex.txt` birleştirildi
- Tüm `class_id` değerleri → `0` (tek sınıf)
- YOLO formatına dönüştürüldü:

```
<class_id> <x_center> <y_center> <width> <height>
```

- %80 Eğitim – %20 Doğrulama ayrımı yapıldı

```
/images/train
/images/val
/labels/train
/labels/val
```

---

## 🧠 Kullanılan Model

YOLOv5s

Avantajları:
- Düşük gecikme
- Hafif mimari
- Gerçek zamanlı kullanım

---

## ⚙️ Eğitim Parametreleri

| Parametre | Değer |
|------------|--------|
| Image Size | 640x640 |
| Epoch | 149 |
| Batch Size | 8 |
| Pretrained | yolov5s.pt |
| Early Stopping | patience = 30 |

---

## 📊 Model Performansı (En İyi Epoch: 84)

| Metrik | Değer |
|--------|--------|
| Precision | 0.911 |
| Recall | 0.950 |
| mAP@0.5 | 0.973 |
| mAP@0.5:0.95 | 0.718 |

Model, sınıf bilgisi olmaksızın tabelaların konumlarını yüksek doğrulukla öğrenmiştir.

---

# 🧠 2️⃣ Trafik Tabelası Sınıflandırma

## Kullanılan Veri Seti

German Traffic Sign Recognition Benchmark (GTSRB)

- 39.209 eğitim görseli
- 12.630 test görseli
- 43 sınıf
- Görseller 32x32 yeniden boyutlandırıldı

---

## 🧩 Veri Hazırlığı

- `.ppm → .jpg` dönüşümü yapıldı
- Görseller normalize edildi
- ML modeller için flatten edildi
- CNN için RGB tensor formatına çevrildi
- 5-Fold Cross Validation (ML modeller)
- %80 – %20 train/val split (CNN)

---

# 🤖 Kullanılan Modeller

## 🥇 CNN (PyTorch)

- Girdi: (3, 32, 32)
- Katmanlar: Conv → ReLU → MaxPool → Dropout → Fully Connected
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Epoch: 20

### 📊 Performans

- Accuracy: 93.56%
- Weighted F1: 0.935

Güçlü olduğu sınıflar:
- 13, 14, 17 (F1 > 0.99)

Zorlanan sınıflar:
- 27 (0.418)
- 30 (0.713)

---

## 🥈 SVM

- Kernel: Linear
- Accuracy: 80.70%

---

## 🥉 Random Forest

- 100 Tree
- Accuracy: 76.17%

---

## 🏅 XGBoost

- objective: multi:softmax
- Accuracy: 76.26%

---

## 📊 Karşılaştırmalı Tablo

| Model | Accuracy | Weighted F1 |
|--------|----------|-------------|
| CNN | 93.56% | 0.935 |
| SVM | 80.70% | 0.809 |
| RF | 76.17% | 0.759 |
| XGBoost | 76.26% | 0.761 |

---

## Kullanılan Teknolojiler

- PyTorch
- OpenCV
- Pillow
- torchvision

---


# 📌 Projenin Teknik Güçlü Yanları

- CRISP-DM uyumlu geliştirme süreci
- Detection + Classification ayrımı
- Model karşılaştırması
- Cross Validation
- Gerçek zamanlı sistem mimarisi
- Web arayüzü entegrasyonu

---

# 🔮 Gelecek Çalışmalar

- Veri artırma (Augmentation)
- Class imbalance düzeltme
- Ensemble model denemeleri
- Embedded sistem entegrasyonu
- Gerçek zamanlı video pipeline optimizasyonu