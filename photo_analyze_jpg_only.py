# photo_analyze_jpg_only.py
# modified to only process JPG files and no RAW
#
#!/usr/bin/env python3
"""
photo_analyze_jpg_only.py

Scans ROOT_DIR for JPG/JPEG images only, extracts and normalizes EXIF datetimes,
tags images (CLIP preferred, torchvision fallback), stores results in DuckDB
using UUID primary keys, and generates summary graphs and a CSV.

Fixes included:
- Restrict to .jpg/.jpeg (case-insensitive)
- Sets PIL.Image.MAX_IMAGE_PIXELS = 933120000
- Normalizes EXIF datetime "YYYY:MM:DD HH:MM:SS" -> "YYYY-MM-DD HH:MM:SS"
- Uses UUID text primary key (DuckDB has no autoincrement)
- CLIP processor uses channels-last numpy arrays with input_data_format
- Pandas explode/dropna fix
"""

from pathlib import Path
from datetime import datetime
import uuid
import sys

# Increase PIL's allowed pixel count to avoid DecompressionBombError
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000

import duckdb
import pandas as pd
from tqdm import tqdm

from PIL import Image as PILImage, ExifTags, ImageStat, ImageOps
import piexif

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional model imports
USE_CLIP = True
USE_TORCHVISION = True
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
except Exception:
    USE_CLIP = False

try:
    import torch
    from torchvision import transforms, models
except Exception:
    USE_TORCHVISION = False

# ---------- Configuration ----------
ROOT_DIR = r"O:\Bilder\1-d7100"
DB_PATH = "photos_db.db"

# Only allow JPG/JPEG files (case-insensitive)
IMAGE_EXTS = {".jpg", ".jpeg"}

COARSE_CATEGORIES = {
    "people": ["person", "man", "woman", "child", "face", "people", "selfie", "group"],
    "birds": ["bird", "crow", "sparrow", "eagle", "duck", "goose", "owl", "hawk", "kite", "seagull"],
    "water": ["sea", "ocean", "lake", "river", "water", "stream", "beach"],
    "forest": ["forest", "wood", "trees", "pine", "jungle"],
    "landscape": ["valley", "mountain", "cliff", "landscape", "field", "meadow", "horizon"],
    "snow": ["snow", "ski", "icy", "blizzard"],
    "spring": ["flower", "blossom", "bloom", "butterfly", "green"],
    "summer": ["sun", "beach", "swim", "sunset"],
    "autumn": ["autumn", "fall", "leaves", "leaf", "pumpkin"]
}

# ---------- Utilities ----------
def norm_text(s):
    return s.strip().lower()

def normalize_exif_datetime(s):
    if not s:
        return None
    s = str(s).strip()
    try:
        dt = datetime.strptime(s, "%Y:%m:%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    try:
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass
    for fmt in ("%Y:%m:%d", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d 00:00:00")
        except Exception:
            continue
    return None

def season_from_iso(dt_iso):
    if not dt_iso:
        return None
    try:
        dt = datetime.strptime(dt_iso, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    m = dt.month
    if m in (12,1,2):
        return "winter"
    if m in (3,4,5):
        return "spring"
    if m in (6,7,8):
        return "summer"
    return "autumn"

def color_temperature_hint(img: PILImage.Image):
    try:
        im = ImageOps.exif_transpose(img).convert("RGB").resize((200,200))
        stat = ImageStat.Stat(im)
        r,g,b = stat.mean
        if g > r and g > b and g > 100:
            return "green"
        if r > 200 and g > 200 and b > 200:
            return "bright"
        if b > r and b > g and b > 120:
            return "blueish"
    except Exception:
        pass
    return None

# ---------- EXIF extraction ----------
def extract_exif(path: Path):
    data = {
        "file_path": str(path),
        "file_name": path.name,
        "datetime_original": None,
        "camera_make": None,
        "camera_model": None,
        "exposure_time": None,
        "f_number": None,
        "iso": None,
        "focal_length": None,
        "width": None,
        "height": None
    }
    try:
        img = PILImage.open(path)
        data["width"], data["height"] = img.size
        try:
            ex = piexif.load(img.info.get("exif", b""))
            zeroth = ex.get("0th", {})
            exif = ex.get("Exif", {})
            def decode_if_bytes(v):
                if isinstance(v, bytes):
                    try:
                        return v.decode(errors="ignore")
                    except:
                        return str(v)
                return v
            if piexif.ImageIFD.Make in zeroth:
                data["camera_make"] = decode_if_bytes(zeroth.get(piexif.ImageIFD.Make))
            if piexif.ImageIFD.Model in zeroth:
                data["camera_model"] = decode_if_bytes(zeroth.get(piexif.ImageIFD.Model))
            if piexif.ExifIFD.DateTimeOriginal in exif:
                dto = decode_if_bytes(exif.get(piexif.ExifIFD.DateTimeOriginal))
                data["datetime_original"] = normalize_exif_datetime(dto)
            if piexif.ExifIFD.ExposureTime in exif:
                data["exposure_time"] = exif.get(piexif.ExifIFD.ExposureTime)
            if piexif.ExifIFD.FNumber in exif:
                data["f_number"] = exif.get(piexif.ExifIFD.FNumber)
            if piexif.ExifIFD.ISOSpeedRatings in exif:
                iso = exif.get(piexif.ExifIFD.ISOSpeedRatings)
                try:
                    data["iso"] = int(iso) if iso is not None else None
                except:
                    data["iso"] = None
            if piexif.ExifIFD.FocalLength in exif:
                data["focal_length"] = exif.get(piexif.ExifIFD.FocalLength)
        except Exception:
            pass
    except Exception as e:
        print(f"[WARN] Could not open {path}: {e}", file=sys.stderr)
    return data

# ---------- Tagging machinery ----------
class Tagger:
    def __init__(self, device=None):
        self.device = device or ("cuda" if 'torch' in sys.modules and torch.cuda.is_available() else "cpu")
        self.use_clip = False
        self.use_torchvision = False
        if USE_CLIP:
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.use_clip = True
            except Exception:
                self.use_clip = False
        if not self.use_clip and USE_TORCHVISION:
            try:
                self.model = models.resnet50(pretrained=True).eval().to(self.device)
                from torchvision import transforms
                self.preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                import json, urllib.request
                try:
                    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                    with urllib.request.urlopen(url, timeout=5) as f:
                        self.imagenet_labels = json.load(f)
                except Exception:
                    self.imagenet_labels = [str(i) for i in range(1000)]
                self.use_torchvision = True
            except Exception:
                self.use_torchvision = False

    def predict_tags(self, image_path: Path, topk=8):
        tags = []
        try:
            img = PILImage.open(image_path).