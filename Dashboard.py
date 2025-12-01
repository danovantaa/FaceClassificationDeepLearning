import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import math
import os
import json 
import cv2 
import mediapipe as mp 

TARGET_SIZE = (224, 224)   # Ubah ke 224x224 agar sesuai dengan input model
MARGIN_RATIO = 0.15 
MAX_DIM = 1600 
MIN_DIM = 256 
USE_FACE_ALIGNMENT = True

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

@st.cache_resource # Gunakan cache agar inisialisasi hanya sekali
def init_mediapipe_models():
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    return face_detection, face_mesh

face_detection, face_mesh = init_mediapipe_models()

def CenterEye(landmarks, image_shape):
    h, w = image_shape[:2]

    # landmark untuk mata kiri dan mata kanan
    EYE_LEFT_IDX = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153]
    EYE_RIGHT_IDX = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380]

    # titik tengah mata kiri
    left_eye_points = []
    for idx in EYE_LEFT_IDX:
        landmark = landmarks.landmark[idx]
        left_eye_points.append([landmark.x * w, landmark.y * h])
    left_eye_center = np.mean(left_eye_points, axis=0)

    # titik tengah mata kanan
    right_eye_points = []
    for idx in EYE_RIGHT_IDX:
        landmark = landmarks.landmark[idx]
        right_eye_points.append([landmark.x * w, landmark.y * h])
    right_eye_center = np.mean(right_eye_points, axis=0)

    return left_eye_center, right_eye_center


def FaceAlign(image, left_eye, right_eye):
    """
    Meluruskan wajah berdasarkan posisi mata kiri dan mata kanan.
    """
    # sudut kemiringan antar mata
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # Titik pusat kedua mata
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)

    #  rotation
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # rotation
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned, angle

def FaceAlignAdvanced(image, landmarks):
    h, w = image.shape[:2]

    # Get eye centers
    left_eye, right_eye = CenterEye(landmarks, image.shape)

    # Landmark hidung
    nose_tip = landmarks.landmark[1]
    nose_point = np.array([nose_tip.x * w, nose_tip.y * h])

    # Calculate the midpoint between eyes
    eyes_center = (left_eye + right_eye) / 2

    # Calculate angle for rotation
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)

    # Rotasi gambar
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned, angle

def Process_Input_Image(pil_img: Image.Image) -> Image.Image | None:
    """
    Deteksi wajah, alignment, dan crop pada satu PIL Image.
    Mengembalikan PIL Image yang siap diproses (224x224).
    """
    
    # 1. Konversi PIL ke CV2 (Numpy Array)
    pil_img = pil_img.convert("RGB")
    # Terapkan EXIF transpose jika ada
    pil_img = ImageOps.exif_transpose(pil_img) 
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Konversi ke BGR (standar CV2)

    orig_img = img.copy()
    orig_h, orig_w = orig_img.shape[:2]

    # 2. Resizing Awal (Logika dari kode Anda)
    h, w = orig_h, orig_w
    if min(h, w) < MIN_DIM:
        scale = MIN_DIM / float(min(h, w))
        img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
        h, w = img.shape[:2]
    
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 3. Alignment Wajah (MediaPipe Face Mesh)
    aligned_img = img.copy()
    if USE_FACE_ALIGNMENT:
        try:
            # MediaPipe membutuhkan RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            results_mesh = face_mesh.process(rgb_img)
            
            if results_mesh.multi_face_landmarks:
                left_eye, right_eye = CenterEye(
                    results_mesh.multi_face_landmarks[0],
                    img.shape
                )
                aligned_img, _ = FaceAlign(img, left_eye, right_eye) # align pada BGR img
            
        except Exception as e:
            st.warning(f"Face Alignment Error: {e}")
            pass # Lanjutkan dengan gambar tanpa alignment

    img = aligned_img
    h, w = img.shape[:2]

    # 4. Deteksi Wajah (MediaPipe Face Detection)
    results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if not results.detections:
        # Coba deteksi pada gambar asli (opsional)
        results = face_detection.process(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        img = orig_img

    if not results.detections:
        # Fallback ke center crop (sesuai logika Anda)
        side = int(0.8 * min(h, w))
        cx, cy = w // 2, h // 2
        x1 = cx - side // 2
        y1 = cy - side // 2
        cropped = img[y1:y1+side, x1:x1+side]
    else:
        # 5. Crop Berdasarkan Bounding Box Terbaik
        best_det = max(results.detections, key=lambda d: d.score[0])
        bbox = best_det.location_data.relative_bounding_box
        
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)

        mx = int(bw * MARGIN_RATIO)
        my = int(bh * MARGIN_RATIO)

        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(x + bw + mx, w)
        y2 = min(y + bh + my, h)

        cropped = img[y1:y2, x1:x2]

    if cropped.size == 0:
        return None

    # 6. Resize Final ke TARGET_SIZE (224x224)
    cropped = cv2.resize(cropped, TARGET_SIZE)

    # 7. Konversi CV2 (BGR) kembali ke PIL Image (RGB)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)

class Configuration:
    data_root: str = "Preprocessed"

    # HYPERPARAMETER
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-3
    label_smoothing = 0.0
    img_size = 224
    batch_size = 64
    num_workers = 0

    # ArcFace hyperparams
    s = 25.0   # Scaling Factor untuk nilai cosine sebelum ke CrossEntropy.
    m = 0.3 # margin sudut

    val_ratio = 0.2
    seed = 42

config = Configuration()
print(config)

class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s = 25.0, # scaling factor
        m = 0.30, # angular margin
        easy_margin: bool = False,
    ):
        super().__init__()

        # Dimensi input (embedding) dan output (jumlah kelas)
        self.in_features = in_features
        self.out_features = out_features

        # Hyperparameter utama ArcFace
        self.s = s # scale
        self.m = m # margin

        # Parameter weight untuk setiap kelas
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) # inisialisasi weight

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # Threshold untuk easy-margin
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # Normalisasi embedding & weight
        cos = F.linear(F.normalize(input), F.normalize(self.weight))

        # Hitung sin(θ) = sqrt(1 - cos²(θ))
        sin = torch.sqrt(1.0 - torch.clamp(cos.pow(2), 0.0, 1.0))

        pi = cos * self.cos_m - sin * self.sin_m  # cos(θ + m) * s

        #Easy margin mode untuk margin handling
        if self.easy_margin:
            pi = torch.where(cos > 0, pi, cos)
        else:
          # adjustment stabilisasi
            pi = torch.where(cos > self.th, pi, cos - self.mm)

        # one-hot label
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * pi) + ((1.0 - one_hot) * cos)
        output *= self.s # scaling factor digunakan sebelum Softmax

        return output

class ResNet100ArcFace(nn.Module):
    """
    Model lengkap:
      - Backbone: IR-100 → embedding 512 dim
      - Head    : ArcMarginProduct (ArcFace) → logits num_classes
    """
    def __init__(self, num_classes: int,
                 embedding_dim: int = 512,
                 s: float = 25.0,
                 m: float = 0.10):
        super().__init__()
        weights = models.ResNet101_Weights.IMAGENET1K_V1
        backbone = models.resnet101(weights=weights)
        embedding_in_features = backbone.fc.in_features

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.embedding_head = nn.Sequential(
            nn.Linear(embedding_in_features, embedding_dim), # Dari 2048 ke 512
            nn.BatchNorm1d(embedding_dim)
        )
        nn.init.constant_(self.embedding_head[1].weight, 1.0)
        nn.init.constant_(self.embedding_head[1].bias, 0.0)
        self.embedding_head[1].bias.requires_grad = False

        self.arc_margin = ArcMarginProduct(
            in_features=embedding_dim, # 512
            out_features=num_classes,
            s=s,
            m=m,
            easy_margin=False,
        )

    def forward(self, x, labels=None):
      x = self.backbone(x)
      # Ekstraksi embedding
      emb = self.embedding_head(x)
      emb = F.normalize(emb, dim=1)
      if labels is None:
          # Inference: hanya cosine * s tanpa margin
          logits = F.linear(
              emb,
              F.normalize(self.arc_margin.weight)
          ) * self.arc_margin.s
      else:
          # Training: pakai ArcFace (cos(theta+m))
          logits = self.arc_margin(emb, labels)

      return logits, emb


MODEL_PATH = "Models\Resnet100ArcFace.pth" 
LABEL_MAP_PATH = "Results\class_labels.json" 
EMBEDDING_DIM = 512

@st.cache_resource
def load_class_labels(path):
    """Memuat dan membalik label map: {indeks: nama_orang}"""
    try:
        with open(path, 'r') as f:
            original_map = json.load(f) 
        
        # Balik map: {0: 'Nama A', 1: 'Nama B', ...}
        return {int(v): k for k, v in original_map.items()} 
    except FileNotFoundError:
        st.error(f"File label map TIDAK DITEMUKAN: {path}")
        return {}
    except Exception as e:
        st.error(f"Error saat memuat label map: {e}")
        return {}

CLASS_LABELS = load_class_labels(LABEL_MAP_PATH)

NUM_CLASSES = len(CLASS_LABELS) 

if NUM_CLASSES == 0:
    st.error("Gagal memuat label kelas. Pastikan file class_labels.json ada dan formatnya benar.")
    st.stop()


# Fungsi untuk memuat model 
@st.cache_resource
def load_trained_model():
    model = ResNet100ArcFace(num_classes=NUM_CLASSES, embedding_dim=EMBEDDING_DIM)
    
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Cek jumlah kelas 
    expected_out = NUM_CLASSES
    actual_out = new_state_dict['arc_margin.weight'].shape[0]
    if expected_out != actual_out:
        st.warning(f"Jumlah kelas tidak cocok! Model memiliki {actual_out} kelas, tetapi {NUM_CLASSES} dimuat.")
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

# Setup Transformasi Inferensi
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def predict_image(model, image, transform, class_labels):
    # Pra-pemrosesan gambar
    img_tensor = transform(image).unsqueeze(0) # unsqueeze(0) menambah dimensi batch
    
    with torch.no_grad():
        # Lakukan inferensi (labels=None agar menggunakan mode inference)
        logits, _ = model(img_tensor, labels=None) 
    
    # Hitung probabilitas menggunakan Softmax
    probabilities = F.softmax(logits, dim=1).squeeze(0)
    
    # Ambil indeks kelas prediksi
    predicted_index = torch.argmax(probabilities).item()
    
    # Ambil probabilitas tertinggi
    max_probability = probabilities[predicted_index].item()
    
    # Dapatkan nama kelas
    predicted_class = class_labels.get(predicted_index, f"Unknown Class Index {predicted_index}")

    return predicted_class, max_probability, probabilities

st.title("Face Recognition Model Dashboard (ResNet-101 + ArcFace)")
st.caption(f"Model dimuat dari: {os.path.basename(MODEL_PATH)}")

# Muat model
model = load_trained_model()

uploaded_file = st.file_uploader("Upload Gambar Wajah untuk Prediksi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Muat gambar yang diunggah
    original_image = Image.open(uploaded_file).convert("RGB")
    
    st.image(original_image, caption='Gambar Asli yang Diunggah', use_container_width=True)
    
    # Tombol Prediksi
    if st.button("Lakukan Prediksi"):
        st.subheader("Hasil Preprocessing dan Prediksi")
        
        processed_image = Process_Input_Image(original_image)
        
        if processed_image is None:
            st.error("Gagal mendeteksi wajah atau menghasilkan gambar yang valid.")
            st.stop()
            
        # Tampilkan gambar yang sudah di-crop dan di-align
        st.image(processed_image, caption=f'Wajah yang Diproses ({TARGET_SIZE[0]}x{TARGET_SIZE[1]})', width=224)
        
        predicted_class, max_probability, probabilities = predict_image(model, processed_image, transform, CLASS_LABELS)
        
        # Tampilkan hasil utama
        st.success(f"**Identitas Prediksi:** {predicted_class}")
        st.info(f"**Probabilitas (Confidence):** {max_probability:.4f}")
        
        # Visualisasi Probabilitas (Top 5)
        st.subheader("Top 5 Probabilitas Kelas")
        
        # Dapatkan Top 5
        top_p, top_class = probabilities.topk(5)
        top_results = []
        for i in range(top_p.size(0)):
            class_name = CLASS_LABELS.get(top_class[i].item(), "Unknown")
            top_results.append({
                "Rank": i + 1,
                "Identitas": class_name,
                "Probabilitas": top_p[i].item()
            })
            
        st.dataframe(top_results, hide_index=True)