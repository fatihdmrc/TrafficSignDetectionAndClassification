import sys
import importlib.util
import subprocess

# Eğer 'pathlib' yanlışlıkla site-packages'ten geliyorsa (3. parti paket),
# onu kaldırıp standart kütüphane pathlib'i kullanacağız.
spec = importlib.util.find_spec("pathlib")
if spec and spec.origin and "site-packages" in spec.origin:
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pathlib"], check=False)

# Bundan sonra güvenle kullanabilirsin
from pathlib import Path
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import io
import sys

# ------------------------------------------------------------
# 0) YOLOv5 KODUNU REPO ICINDEN IMPORT EDEBILMEK ICIN
#    - yolov5 klasorunu Python path'ine ekliyoruz.
#    - Böylece: from models.common import DetectMultiBackend gibi importlar çalışır.
# ------------------------------------------------------------
YOLOV5_DIR = Path(__file__).resolve().parent / "yolov5"
sys.path.append(str(YOLOV5_DIR))

# YOLOv5'in kendi modulleri
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
#    - .pth weight dosyani yuklemek icin mimariyi ayni sekilde tanimlamaliyiz.
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
# 3) CNN ICIN DONUSTURME (32x32 + normalize)
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ------------------------------------------------------------
# 4) YOLOv5 ICIN ON-ISLEME FONKSIYONU
#    - YOLOv5 modeline vermek icin gorseli:
#      BGR->RGB, yeniden boyutlandirma, normalizasyon, tensor'a cevirme yapacagiz.
#    - Letterbox (oran koruyarak padding) YOLOv5 icinde var ama DetectMultiBackend'e
#      dogrudan "raw image" gondermek yerine tensor hazirlayacagiz.
# ------------------------------------------------------------
def preprocess_for_yolo(img_bgr, img_size=640):
    """
    img_bgr: OpenCV BGR goruntu
    img_size: YOLO input boyutu (genelde 640)
    return:
      img: torch tensor (1,3,H,W) [0..1] float32
      img0: orijinal bgr (cizim icin)
    """
    img0 = img_bgr.copy()

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # YOLOv5 letterbox fonksiyonu utils.augmentations icinde
    # Import'u burada yapiyoruz ki başlangıçta gereksiz yük olmasın.
    from utils.augmentations import letterbox

    # Oran koruyarak pad ile 640'a getir
    img_lb = letterbox(img_rgb, new_shape=img_size, auto=False)[0]

    # HWC -> CHW
    img_chw = img_lb.transpose((2, 0, 1))

    # contiguous array (PyTorch icin hizli bellek)
    img_chw = np.ascontiguousarray(img_chw)

    # numpy -> torch
    img = torch.from_numpy(img_chw).float()

    # 0-255 -> 0-1
    img /= 255.0

    # batch dimension ekle (1,3,H,W)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img, img0, img_lb.shape[:2]  # (H,W) of letterboxed

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

    # CPU kullanacagiz (Streamlit Cloud genelde GPU vermez)
    device = select_device("cpu")

    # YOLOv5 backend (pt/onnx vb. destekler)
    yolo_model = DetectMultiBackend(str(yolo_weights), device=device)
    yolo_model.eval()

    # CNN modeli
    cnn_model = CNNModel()
    cnn_model.load_state_dict(torch.load(str(cnn_weights), map_location="cpu"))
    cnn_model.eval()

    return yolo_model, cnn_model

# ------------------------------------------------------------
# 6) ANA ISLEME: YOLO tespit -> crop -> CNN siniflandir -> ciz
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
    """
    image_pil: PIL Image (RGB)
    yolo_conf_threshold: YOLO tespit guven esigi
    yolo_iou_threshold : NMS icin IOU esigi
    cnn_conf_threshold : CNN siniflandirma guven esigi
    img_size: YOLO input boyutu (genelde 640)
    """

    # PIL -> numpy RGB
    img_rgb = np.array(image_pil).astype(np.uint8)

    # Cok buyuk goruntuyu kucult (server bellek/performans icin)
    if img_rgb.shape[1] > max_width:
        scale = max_width / img_rgb.shape[1]
        new_size = (int(img_rgb.shape[1] * scale), int(img_rgb.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, new_size)

    # Cizim icin BGR
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ---------------------------
    # YOLO ON-ISLEME
    # ---------------------------
    img_tensor, img0_bgr, lb_hw = preprocess_for_yolo(img_bgr, img_size=img_size)

    # Modelin cihazina tasi (cpu)
    img_tensor = img_tensor.to(yolo_model.device)

    # ---------------------------
    # YOLO INFERENCE
    # ---------------------------
    with torch.no_grad():
        pred = yolo_model(img_tensor)

    # ---------------------------
    # NMS (Non-Maximum Suppression)
    # - Aynı nesneye ait üst üste kutuları eler
    # ---------------------------
    pred = non_max_suppression(
        prediction=pred,
        conf_thres=float(yolo_conf_threshold),
        iou_thres=float(yolo_iou_threshold),
        classes=None,      # sinif filtreleme yok
        agnostic=False,
        max_det=1000
    )

    det = pred[0]  # tek görsel

    detections_for_table = []

    if det is None or len(det) == 0:
        st.warning("YOLO modeli gorselde herhangi bir trafik tabelasi bulamadi.")
        return img0_bgr, detections_for_table

    # det formatı: (N, 6) -> [x1,y1,x2,y2,conf,cls]
    # Letterbox uygulanmış koordinatları orijinal görüntüye ölçekleyeceğiz.
    # img_tensor'ın boyutu -> letterbox boyutu
    img_h, img_w = img_tensor.shape[2], img_tensor.shape[3]

    # Orijinal (yeniden boyutlanmış) görüntü boyutu
    h0, w0 = img0_bgr.shape[:2]

    # Kutuları orijinale ölçekle
    det[:, :4] = scale_boxes((img_h, img_w), det[:, :4], (h0, w0)).round()

    # ---------------------------
    # Her tespit icin crop + CNN
    # ---------------------------
    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        yolo_conf = float(conf)
        yolo_cls = int(cls)

        # sınır düzelt
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)

        cropped_rgb = img_rgb[y1:y2, x1:x2]
        if cropped_rgb.size == 0:
            continue

        # CNN girişi
        pil_crop = Image.fromarray(cropped_rgb)
        input_tensor = transform(pil_crop).unsqueeze(0)

        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            cnn_conf, pred_cls = torch.max(probs, dim=1)

        cnn_conf = float(cnn_conf.item())
        pred_cls = int(pred_cls.item())

        # CNN esigi
        if cnn_conf < float(cnn_conf_threshold):
            continue

        label_name = class_names[pred_cls] if 0 <= pred_cls < len(class_names) else f"Sinif {pred_cls}"
        label = f"{label_name} ({cnn_conf:.2f})"

        # Cizim
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
        st.info("Kontrol: yolov5/ klasoru repo icinde mi? models/best.pt ve models/gtsrb_cnn_model.pth var mi?")