# 🚦 Trafik Tabelası Tespiti Projesi  
## Canlıya Alma ve Sorun Çözüm Süreci

Bu doküman, **YOLOv5 + CNN tabanlı trafik tabelası tespit ve sınıflandırma sisteminin** Streamlit Cloud ortamına taşınması sırasında yaşanan teknik sorunları ve uygulanan çözümleri adım adım açıklamaktadır.

---

## 📌 1. Projenin Amacı

Bu proje:

- Görsel içindeki trafik tabelalarını **YOLOv5** ile tespit eder.
- Tespit edilen tabelaları **CNN modeli** ile sınıflandırır.
- Sonuçları **Streamlit arayüzü** üzerinden kullanıcıya gösterir.
- İşlenmiş görselin indirilmesine olanak sağlar.

---

## 🚀 2. Projenin Canlı Ortama Taşınması

### 🔧 Yapılan İşlemler

1. Kodlar GitHub’a yüklendi.
2. Model dosyaları projeye eklendi:
   - `models/best.pt` → YOLOv5 modeli
   - `models/gtsrb_cnn_model.pth` → CNN modeli
3. Streamlit Cloud üzerinden deploy işlemi yapıldı.

### 📌 İlk Durum

- Arayüz başarıyla açıldı.
- Ancak görsel yüklenince çeşitli hatalar oluştu.

---

## 🧩 3. Eksik Kütüphane Sorunları

### ❌ Alınan Hatalar

```bash
No module named 'ultralytics'
No module named 'tqdm'
No module named 'seaborn' 
```
## ✅ Çözüm

- Tüm gerekli bağımlılıklar requirements.txt dosyasına eklendi.

- Ortam yeniden başlatıldı.
## 🖥️ 4. OpenCV (cv2) Hatası
### ❌ Alınan Hata

- import cv2 sırasında hata oluştu.

### 🤔 Neden?

- Streamlit Cloud grafik arayüz (GUI) içermediği için standart OpenCV paketi çalışmadı.

### ✅ Çözüm

- Standart OpenCV yerine:

- opencv-python-headless

paketi kullanıldı.

Ayrıca Python sürümü 3.11 olarak sabitlendi.
## 🔄 5. YOLO Model Format Uyumsuzluğu
### ❌ Alınan Hata
- No module named 'models.yolo'

### 🤔 Neden?

- Model dosyası YOLOv5 ile eğitilmişti ancak farklı bir API ile yüklenmeye çalışıldı.

### ✅ Çözüm

- YOLOv5 GitHub kodu doğrudan proje içine eklendi.

- Model internetten çekilmeden, lokal YOLOv5 kodu ile çalıştırıldı.

## ⚙️ 6. Ultralytics Otomatik Kurulum Sorunu

- YOLOv5 kodunun içinde şu yapı vardı:

- pip install ultralytics


Bu yapı canlı ortamda dinamik paket kurulumu yapmaya çalışıyordu.

### ❌ Sonuç

- Bağımlılık zinciri oluştu.

- Ortam karışıklığı meydana geldi.

### ✅ Çözüm

- Otomatik kurulum devre dışı bırakıldı.

- Gerekli paketler sabit şekilde requirements.txt dosyasına eklendi.

## ❗ 7. En Kritik Sorun: pathlib._local Hatası
### ❌ Alınan Hata
- No module named 'pathlib._local'; 'pathlib' is not a package

### 🤔 Neden?

- Model dosyası Windows ortamında kaydedilmişti.

- Canlı sunucu Linux ortamında çalışıyordu.

- Dosya yolu (path) sistemleri arasında uyumsuzluk oluştu.

### ✅ Çözüm

- app.py dosyasının en üstüne sistem uyumluluk kodu eklendi:

- pathlib yönlendirmesi yapıldı.

- WindowsPath ve PosixPath eşlemesi sağlandı.

Bu sayede model dosyası Linux ortamında sorunsuz şekilde yüklendi.