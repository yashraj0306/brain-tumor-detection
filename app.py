import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import segmentation_models_pytorch as smp
import torch.nn as nn
import base64

# ── Model definition ─────────────────────────────────────────────────────────
class MultiTaskUNet(nn.Module):
    def __init__(self, num_seg_classes=1, num_cls_classes=4):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            "efficientnet-b3",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )

        self.decoder = smp.decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_norm='batchnorm',
            center=False,
            attention_type="scse"
        )

        self.seg_head = smp.base.SegmentationHead(
            in_channels=16,
            out_channels=num_seg_classes,
            activation=None,
            kernel_size=3
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_cls_classes)
        )

    def forward(self, x):
        features    = self.encoder(x)
        decoder_out = self.decoder(features)
        seg_mask    = self.seg_head(decoder_out)
        cls_out     = self.cls_head(features[-1])
        return seg_mask, cls_out

# ── Load model ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

model = MultiTaskUNet(num_seg_classes=1, num_cls_classes=4)
model.load_state_dict(torch.load("Best_Model/best_model_v2.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI()

@app.get("/")
def index():
    return FileResponse("index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img.resize((256, 256))

    # Preprocessing — ImageNet normalization (must match training)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = (img_array - mean) / std
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        seg_out, cls_out = model(img_tensor)

    # Classification
    cls_probs  = torch.softmax(cls_out, dim=1)[0].cpu().numpy()
    cls_idx    = int(np.argmax(cls_probs))
    cls_label  = CLASS_NAMES[cls_idx]
    confidence = float(cls_probs[cls_idx]) * 100

    # Binary segmentation mask
    seg_prob    = torch.sigmoid(seg_out)[0, 0].cpu().numpy()  # (256, 256)
    binary_mask = (seg_prob > 0.5).astype(np.uint8)

    # Overlay
    overlay = np.array(img_resized).copy()
    if cls_label != "no_tumor" and binary_mask.sum() > 0:
        color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
        color_mask[binary_mask == 1] = [255, 0, 0]  # red
        overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

    # Encode to base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "label": cls_label,
        "confidence": round(confidence, 2),
        "overlay": overlay_b64,
        "all_probs": {CLASS_NAMES[i]: round(float(cls_probs[i]) * 100, 2) for i in range(4)}
    })