# ============================================================
# PATHLIB UYUMLULUK YAMASI (EN USTTE OLMALI!)
# - Bazi ortamlar/weight dosyalari 'pathlib._local' arayabilir.
# - Linux'ta WindowsPath/PosixPath uyumsuzlugu da cikabilir.
# ============================================================
import os
import sys
import pathlib

# 'pathlib._local' import ediliyorsa stdlib pathlib'e yönlendir
sys.modules["pathlib._local"] = pathlib

# Linux'ta WindowsPath nesnesi pickle'dan cikarsa patlayabilir.
# Bu yüzden WindowsPath'i PosixPath'e esliyoruz.
if os.name != "nt":
    pathlib.WindowsPath = pathlib.PosixPath
# ============================================================

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import io
from pathlib import Path  # artık güvenli

# ------------------------------------------------------------
# 0) YOLOv5 KODUNU REPO ICINDEN IMPORT EDEBILMEK ICIN
#    - yolov5 klasorunu Python path'ine ekliyoruz.
#    - Böylece: from models.common import DetectMultiBackend gibi importlar çalışır.
# ------------------------------------------------------------
YOLOV5_DIR = Path(__file__).resolve().parent / "yolov5"
sys.path.append(str(YOLOV5_DIR))

# YOLOv5 modulleri (repo icinden)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Trafik Tabelasi Tanima (YOLOv5 + CNN)",
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
# 4) YOLOv5 ICIN ON-ISLEME (LETTERBOX)
# ------------------------------------------------------------
def preprocess_for_yolo(img_bgr, img_size=640):
    """
    img_bgr: OpenCV BGR goruntu
    img_size: YOLO input boyutu (genelde 640)

    Dönenler:
      img_tensor: (1,3,H,W) float [0..1]
      img0_bgr   : orijinal (cizim icin)
      img_hw     : model girdisinin (H,W)
    """
    img0_bgr = img_bgr.copy()

    # BGR -> RGB (YOLOv5 genelde RGB ile calisir)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Letterbox: oran koruyarak padding ile istenen boyuta getirir
    from utils.augmentations import letterbox
    img_lb = letterbox(img_rgb, new_shape=img_size, auto=False)[0]

    # HWC -> CHW
    img_chw = img_lb.transpose((2, 0, 1))
    img_chw = np.ascontiguousarray(img_chw)

    # numpy -> torch
    img_tensor = torch.from_numpy(img_chw).float()
    img_tensor /= 255.0  # 0-255 -> 0-1

    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, img0_bgr, (img_lb.shape[0], img_lb.shape[1])

# ------------------------------------------------------------
# 5) MODELLERI YUKLE (YOLOv5 + CNN)
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    yolo_weights = Path("models/best.pt")
    cnn_weights = Path("models/gtsrb_cnn_model.pth")

    if not yolo_weights.exists():
        raise FileNotFoundError(f"YOLO model bulunamadi: {yolo_weights.resolve()}")
    if not cnn_weights.exists():
        raise FileNotFoundError(f"CNN model bulunamadi: {cnn_weights.resolve()}")

    # Streamlit Cloud genelde CPU
    device = select_device("cpu")

    # YOLOv5 backend (pt dosyasini torch.load ile yukler)
    yolo_model = DetectMultiBackend(str(yolo_weights), device=device)
    yolo_model.eval()

    # CNN modeli
    cnn_model = CNNModel()
    cnn_model.load_state_dict(torch.load(str(cnn_weights), map_location="cpu"))
    cnn_model.eval()

    return yolo_model, cnn_model

# ------------------------------------------------------------
# 6) ANA ISLEME: YOLO tespit -> crop -> CNN -> ciz
# ------------------------------------------------------------
def process_image(
    image_pil,
    yolo_model,
    cnn_model,
    yolo_conf_threshold=0.5,
    yolo_iou_threshold=0.45,
    cnn_conf_threshold=0.7,
    img_size=640,
    max_width=1280
):
    # PIL -> numpy RGB
    img_rgb = np.array(image_pil).astype(np.uint8)

    # Cok buyuk goruntuyu kucult (performans/bellek)
    if img_rgb.shape[1] > max_width:
        scale = max_width / img_rgb.shape[1]
        new_size = (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size)

    # Cizim icin BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # YOLO on-isleme
    img_tensor, img0_bgr, _ = preprocess_for_yolo(img_bgr, img_size=img_size)
    img_tensor = img_tensor.to(yolo_model.device)

    # YOLO inference
    with torch.no_grad():
        pred = yolo_model(img_tensor)

    # NMS
    pred = non_max_suppression(
        prediction=pred,
        conf_thres=float(yolo_conf_threshold),
        iou_thres=float(yolo_iou_threshold),
        classes=None,
        agnostic=False,
        max_det=300
    )

    det = pred[0]  # tek goruntu
    detections_for_table = []

    if det is None or len(det) == 0:
        st.warning("YOLO modeli gorselde herhangi bir trafik tabelasi bulamadi.")
        return img0_bgr, detections_for_table

    # Letterbox boyutu: img_tensor (H,W)
    img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]
    # Orijinal (yeniden boyutlanmis) goruntu boyutu
    h0, w0 = img0_bgr.shape[:2]

    # Kutuları orijinal goruntu boyutuna ölçekle
    det[:, :4] = scale_boxes((img_h, img_w), det[:, :4], (h0, w0)).round()

    # Her tespit
    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        yolo_conf = float(conf)
        yolo_cls = int(cls)

        # Sinir kontrolü
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)

        crop_rgb = img_rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            continue

        # CNN sınıflandırma
        pil_crop = Image.fromarray(crop_rgb)
        input_tensor = transform(pil_crop).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            cnn_conf, pred_cls = torch.max(probs, dim=1)

        cnn_conf = float(cnn_conf.item())
        pred_cls = int(pred_cls.item())

        if cnn_conf < float(cnn_conf_threshold):
            continue

        label_name = class_names[pred_cls] if 0 <= pred_cls < len(class_names) else f"Sinif {pred_cls}"
        label = f"{label_name} ({cnn_conf:.2f})"

        # Çizim
        cv2.rectangle(img0_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img0_bgr,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        detections_for_table.append({
            "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
            "yolo_conf": round(yolo_conf, 3),
            "yolo_cls": yolo_cls,
            "cnn_class": pred_cls,
            "cnn_label": label_name,
            "cnn_conf": round(cnn_conf, 3),
        })

    return img0_bgr, detections_for_table

# ------------------------------------------------------------
# 7) STREAMLIT UI
# ------------------------------------------------------------
st.title("🚦 Trafik Tabelasi Tanima (YOLOv5 + CNN)")
st.write("Gorsel yukleyin → YOLOv5 tespit etsin → CNN siniflandirsin.")

with st.sidebar:
    st.header("Ayarlar")
    yolo_conf = st.slider("YOLO guven esigi (conf)", 0.05, 0.95, 0.50, 0.05)
    yolo_iou = st.slider("YOLO NMS IOU esigi", 0.10, 0.90, 0.45, 0.05)
    cnn_conf = st.slider("CNN guven esigi", 0.05, 0.99, 0.70, 0.01)
    img_size = st.selectbox("YOLO giris boyutu", [640, 512, 416], index=0)
    max_w = st.selectbox("Maksimum goruntu genisligi", [640, 800, 1024, 1280], index=3)

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
            yolo_iou_threshold=float(yolo_iou),
            cnn_conf_threshold=float(cnn_conf),
            img_size=int(img_size),
            max_width=int(max_w),
        )

        st.subheader("Tespit Edilen Tabelalar (YOLO + CNN)")
        st.dataframe(detections, use_container_width=True)

        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        st.image(processed_rgb, caption="Islenmis Gorsel", use_container_width=True)

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
        st.info("Kontrol: yolov5/ klasoru repo icinde mi? models/best.pt ve models/gtsrb_cnn_model.pth var mi?")
