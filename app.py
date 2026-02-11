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

# YOLO'yu torch.hub ile degil, direkt Ultralytics API ile kullanacagiz
from ultralytics import YOLO

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Trafik Tabelasi Tanima (YOLO + CNN)",
    page_icon="🚦",
    layout="wide"
)

# ------------------------------------------------------------
# 1) CNN MODEL MIMARISI
# ------------------------------------------------------------
class CNNModel(nn.Module):
    def __init__(self, num_classes=43):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------------------------------------------------
# 2) SINIF ISIMLERI (43)
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
# 3) CNN ICIN DONUSTURME
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------------------------------------------------
# 4) MODELLERI YUKLE
#    - YOLO: Ultralytics YOLO API
#    - CNN: PyTorch state_dict
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    yolo_model_path = Path("models/best.pt")
    cnn_model_path = Path("models/gtsrb_cnn_model.pth")

    if not yolo_model_path.exists():
        raise FileNotFoundError(f"YOLO modeli bulunamadi: {yolo_model_path.resolve()}")
    if not cnn_model_path.exists():
        raise FileNotFoundError(f"CNN modeli bulunamadi: {cnn_model_path.resolve()}")

    # YOLO modelini yukle (custom weights)
    # Bu satir torch.hub gibi repo indirmez; cok daha stabil.
    yolo_model = YOLO(str(yolo_model_path))

    # CNN modelini yukle
    cnn_model = CNNModel()
    cnn_model.load_state_dict(torch.load(str(cnn_model_path), map_location="cpu"))
    cnn_model.eval()

    return yolo_model, cnn_model

# ------------------------------------------------------------
# 5) GORSEL ISLEME: YOLO tespit -> crop -> CNN siniflandirma -> ciz
# ------------------------------------------------------------
def process_image(image_pil, yolo_model, cnn_model, yolo_conf_threshold=0.5, cnn_conf_threshold=0.7, max_width=640):
    # PIL -> NumPy RGB
    img_rgb = np.array(image_pil).astype(np.uint8)

    # Büyük görselleri küçült (performans/bellek)
    if img_rgb.shape[1] > max_width:
        scale = max_width / img_rgb.shape[1]
        new_size = (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size)

    # Çizim için OpenCV BGR kopyası
    image_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ---------------------------
    # YOLO inference
    # ---------------------------
    # conf parametresi: YOLO'nun tespit guven esigi
    results = yolo_model.predict(source=img_rgb, conf=float(yolo_conf_threshold), verbose=False)

    # Ultralytics'te ilk frame/tek image icin results[0]
    res0 = results[0]
    boxes = res0.boxes  # Boxes nesnesi (xyxy, conf, cls)

    if boxes is None or len(boxes) == 0:
        st.warning("YOLO modeli gorselde herhangi bir trafik tabelasi bulamadi.")
        return image_cv, None

    # Streamlit'te tablo gostermek istersen diye bir liste olusturalim
    detections_for_table = []

    # ---------------------------
    # Kutulari tek tek gez
    # ---------------------------
    for box in boxes:
        # xyxy koordinatlari
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # YOLO confidence
        yolo_conf = float(box.conf[0].cpu().numpy())

        # YOLO class id (bu projede YOLO sadece "tabela" tespit ediyor olabilir)
        # eger YOLO birden fazla sinifla egitildiyse burada cls anlamli olur
        yolo_cls = int(box.cls[0].cpu().numpy()) if box.cls is not None else -1

        # Sınır kontrolü
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_cv.shape[1], x2), min(image_cv.shape[0], y2)

        cropped = img_rgb[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # CNN siniflandirma
        pil_crop = Image.fromarray(cropped)
        input_tensor = transform(pil_crop).unsqueeze(0)  # (1, 3, 32, 32)

        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            cnn_conf, predicted_class = torch.max(probs, dim=1)

        cnn_conf = float(cnn_conf.item())
        predicted_class = int(predicted_class.item())

        # CNN guven esiginin altindakileri gec
        if cnn_conf < float(cnn_conf_threshold):
            continue

        # Etiket metni
        class_name = class_names[predicted_class] if 0 <= predicted_class < len(class_names) else f"Sinif {predicted_class}"
        label = f"{class_name} ({cnn_conf:.2f})"

        # Kutuyu ciz + yaziyi yaz
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

        # Tablo icin kayit
        detections_for_table.append({
            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
            "yolo_conf": round(yolo_conf, 3),
            "yolo_cls": yolo_cls,
            "cnn_class": predicted_class,
            "cnn_label": class_name,
            "cnn_conf": round(cnn_conf, 3)
        })

    return image_cv, detections_for_table

# ------------------------------------------------------------
# 6) STREAMLIT UI
# ------------------------------------------------------------
st.title("🚦 Trafik Tabelasi Tanima (YOLO + CNN)")
st.write("Gorsel yukleyin → YOLO tabelalari tespit etsin → CNN tabelayi siniflandirsin.")

with st.sidebar:
    st.header("Ayarlar")
    yolo_conf = st.slider("YOLO guven esigi (conf)", 0.05, 0.95, 0.50, 0.05)
    cnn_conf = st.slider("CNN guven esigi", 0.05, 0.99, 0.70, 0.01)
    max_w = st.selectbox("Maksimum goruntu genisligi", [640, 800, 1024, 1280], index=0)

uploaded_file = st.file_uploader("Bir trafik sahnesi gorseli yukleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yuklenen Gorsel", use_container_width=True)

        yolo_model, cnn_model = load_models()

        processed_bgr, detections = process_image(
            image_pil=image,
            yolo_model=yolo_model,
            cnn_model=cnn_model,
            yolo_conf_threshold=float(yolo_conf),
            cnn_conf_threshold=float(cnn_conf),
            max_width=int(max_w)
        )

        if detections is not None:
            st.subheader("Tespit Edilen Tabelalar (YOLO + CNN)")
            st.dataframe(detections, use_container_width=True)

        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        st.image(processed_rgb, caption="Islenmis Gorsel (Kutular + CNN Etiketleri)", use_container_width=True)

        # Indirme
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
        st.info("Kontrol: models/best.pt ve models/gtsrb_cnn_model.pth repo icinde mevcut mu?")