import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import io
from pathlib import Path

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Trafik Tabelasi Tanima (YOLO + CNN)",
    page_icon="🚦",
    layout="wide"
)

# ------------------------------------------------------------
# 1) CNN MODEL MİMARİSİ
#    - .pth dosyan sadece agirliklari (weight) tutar.
#    - Bu nedenle ayni mimariyi burada tanimlamak zorundayiz.
# ------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=43):
        super(CNNModel, self).__init__()

        # 1. evrişim katmani (Convolution)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        # 2. evrişim katmani
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # Maksimum havuzlama (Max Pooling): boyut azaltma
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Asiri ogrenmeyi (overfitting) azaltmak icin dropout
        self.dropout = nn.Dropout(0.25)

        # Tam bagli katmanlar (Fully Connected)
        # Not: 32x32 giris -> conv/pool islemlerinden sonra 64 * 6 * 6 olacagi varsayilir
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # ReLU aktivasyon + pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Tensor'u duzlestir (flatten)
        x = x.view(-1, 64 * 6 * 6)

        # FC1 + ReLU + dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Cikis katmani (logit)
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# 2) SINIF İSİMLERİ (GTSRB - 43 sinif)
# ------------------------------------------------------------
class_names = [
    "Hiz limiti 20", "Hiz limiti 30", "Hiz limiti 50", "Hiz limiti 60",
    "Hiz limiti 70", "Hiz limiti 80", "Hiz limiti sonu 80", "Hiz limiti 100",
    "Hiz limiti 120", "Sollama yasagi", "3.5 ton uzeri sollama yasagi",
    "Ilk gecis hakki", "Ana yol", "Yol ver", "Dur",
    "Arac giremez", "3.5 ton uzeri arac giremez", "Giris yasak", "Genel tehlike",
    "Sola viraj", "Saga viraj", "S-viraj", "Kasisli yol",
    "Kaygan yol", "Sagdan daralan yol", "Yol calismasi", "Trafik lambasi", "Yaya gecidi",
    "Cocuk gecidi", "Bisiklet gecidi", "Buzlanma", "Hayvan gecisi",
    "Tum sinirlamalarin sonu", "Saga don", "Sola don",
    "Sadece ileri", "Ileri veya saga", "Ileri veya sola", "Sagdan gidiniz", "Soldan gidiniz",
    "Donel kavsak", "Sollama yasagi sonu", "3.5 ton uzeri sollama yasagi sonu"
]

# ------------------------------------------------------------
# 3) CNN ICIN GORSEL DONUSTURME (32x32 + normalize)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CNN girdisi 32x32
    transforms.ToTensor(),        # PIL -> Tensor [0..1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
])

# ------------------------------------------------------------
# 4) MODELLERİ YÜKLEME
#    - st.cache_resource: Streamlit tekrar calistirmalarda yeniden yuklemeyi engeller.
#    - Modeller repo icindeki models/ klasorunden yuklenir.
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    # Repo içi yollar (relative path)
    yolo_model_path = Path("models/best.pt")
    cnn_model_path = Path("models/gtsrb_cnn_model.pth")

    # Dosyalar var mi kontrol edelim (canli ortamda debug kolay olur)
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO model dosyasi bulunamadi: {yolo_model_path.resolve()}")
    if not cnn_model_path.exists():
        raise FileNotFoundError(f"CNN model dosyasi bulunamadi: {cnn_model_path.resolve()}")

    # YOLOv5 custom modeli yukle
    # torch.hub: ultralytics/yolov5 deposunu indirip cache'e alir.
    # Streamlit Cloud'da ilk acilis biraz uzun surebilir (normal).
    yolo_model = torch.hub.load(
        repo_or_dir="ultralytics/yolov5",
        model="custom",
        path=str(yolo_model_path),
        force_reload=False
    )

    # YOLO icin guven esigi (detection confidence threshold)
    yolo_model.conf = 0.5

    # CNN modeli yukle
    cnn_model = CNNModel()
    cnn_model.load_state_dict(torch.load(str(cnn_model_path), map_location="cpu"))
    cnn_model.eval()  # inference moduna al

    return yolo_model, cnn_model

# ------------------------------------------------------------
# 5) GORSEL ISLEME: YOLO tespit -> crop -> CNN siniflandirma -> cizim
# ------------------------------------------------------------
def process_image(image_pil, yolo_model, cnn_model, cnn_conf_threshold=0.70, max_width=640):
    """
    image_pil: Yuklenen PIL goruntu
    yolo_model: YOLOv5 tespit modeli
    cnn_model: CNN siniflandirma modeli
    cnn_conf_threshold: CNN siniflandirma guven esigi
    max_width: Çok buyuk goruntuleri kisaltmak icin max genislik
    """

    # PIL -> NumPy RGB (uint8)
    img_rgb = np.array(image_pil).astype(np.uint8)

    # Çok buyuk goruntuyu kucult (bellek/perf)
    if img_rgb.shape[1] > max_width:
        scale = max_width / img_rgb.shape[1]
        new_size = (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size)

    # Cizim icin OpenCV BGR kopyası
    image_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # YOLO inference (RGB numpy giris kabul eder)
    results = yolo_model(img_rgb)
    results_pd = results.pandas().xyxy[0]  # pandas dataframe

    # Hic tespit yoksa uyar
    if results_pd.empty:
        st.warning("YOLO modeli gorselde herhangi bir trafik tabelasi bulamadi.")
        return image_cv, results_pd

    # Her tespit icin crop + CNN siniflandirma
    for _, row in results_pd.iterrows():
        # Kutucuk koordinatlarini al
        x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])

        # Sinirlar disina cikmayalim
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_cv.shape[1], x2), min(image_cv.shape[0], y2)

        # Crop al
        cropped = img_rgb[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # Crop'u CNN girisine hazirla
        pil_crop = Image.fromarray(cropped)
        input_tensor = transform(pil_crop).unsqueeze(0)  # (1,3,32,32)

        with torch.no_grad():
            output = cnn_model(input_tensor)                 # logit
            probs = torch.softmax(output, dim=1)            # olasilik
            conf, pred = torch.max(probs, dim=1)            # en iyi sinif

        confidence = float(conf.item())
        predicted_class = int(pred.item())

        # CNN guven esiginin altindakileri cizme
        if confidence < cnn_conf_threshold:
            continue

        # Etiket metni
        class_name = class_names[predicted_class] if 0 <= predicted_class < len(class_names) else f"Sinif {predicted_class}"
        label = f"{class_name} ({confidence:.2f})"

        # Kutuyu ve etiketi goruntuye ciz
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image_cv,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return image_cv, results_pd

# ------------------------------------------------------------
# 6) STREAMLIT ARAYUZ
# ------------------------------------------------------------
st.title("🚦 Trafik Tabelasi Tanima (YOLO + CNN)")
st.write("Gorsel yukleyin → YOLO tespit etsin → her tabelayi CNN siniflandirsin.")

# Kullanici ayarlari (slider)
with st.sidebar:
    st.header("Ayarlar")
    yolo_conf = st.slider("YOLO guven esigi (conf)", 0.05, 0.95, 0.50, 0.05)
    cnn_conf = st.slider("CNN guven esigi", 0.05, 0.99, 0.70, 0.01)
    max_w = st.selectbox("Maksimum goruntu genisligi", [640, 800, 1024, 1280], index=0)

uploaded_file = st.file_uploader("Bir trafik sahnesi gorseli yukleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Goruntuyu oku
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yuklenen Gorsel", use_container_width=True)

        # Modelleri yukle
        yolo_model, cnn_model = load_models()

        # Sidebar'dan YOLO conf ayarini modele uygula
        yolo_model.conf = float(yolo_conf)

        # Isle
        processed_bgr, detections_df = process_image(
            image_pil=image,
            yolo_model=yolo_model,
            cnn_model=cnn_model,
            cnn_conf_threshold=float(cnn_conf),
            max_width=int(max_w)
        )

        # Tespit tablosu
        st.subheader("YOLO Tespit Sonuclari")
        st.dataframe(detections_df, use_container_width=True)

        # OpenCV(BGR) -> Streamlit(RGB)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        st.image(processed_rgb, caption="Islenmis Gorsel (Kutular + CNN Etiketleri)", use_container_width=True)

        # Indirme butonu
        result_img = Image.fromarray(processed_rgb)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="Islenmis Gorseli Indir",
            data=buf.getvalue(),
            file_name="sonuc.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Hata olustu: {e}")
        st.info("Ipucu: models/ klasorunde best.pt ve gtsrb_cnn_model.pth dosyalari oldugundan emin olun.")
