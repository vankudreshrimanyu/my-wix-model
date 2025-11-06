import os, io
from typing import Optional
from PIL import Image
import numpy as np
import torch, torch.nn as nn
import timm
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

# ---------- Config ----------
MODEL_PATH = os.getenv("MODEL_PATH", "xception_torch_best.pt")
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Allow your Wix domain (edit if you add a custom domain later)
ALLOWED_ORIGINS = [
    "https://xdk0aj-my-site-ia9z7xqu-vankudreshrimanyu.wix-vibe.com",
    "https://www.wix.com",
]

# ---------- App ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Model ----------
sigmoid = nn.Sigmoid()

def build_model():
    backbone = timm.create_model("legacy_xception", pretrained=False, num_classes=0)
    in_feats = backbone.num_features
    head = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_feats, 1))
    model = nn.Sequential(backbone, head)

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model

model = build_model()

# ---------- Preprocess (ImageNet) ----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    return tensor

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = 0.5):
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content))
        x = preprocess(img)
        with torch.no_grad():
            logit = model(x)              # shape [1,1]
            prob_fake = float(sigmoid(logit).squeeze(1).item())

        label = "fake" if prob_fake >= float(threshold) else "real"
        return JSONResponse({
            "prob_fake": prob_fake,
            "label": label,
            "threshold": float(threshold)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
