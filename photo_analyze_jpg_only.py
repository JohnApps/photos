# photo_analyze_jpg_only.py
# modified to only process JPG files and no RAW
#
#!/usr/bin/env python3
<<<<<<< HEAD
=======
#!/usr/bin/env python3
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
"""
photo_analyze_jpg_only.py

Scans ROOT_DIR for JPG/JPEG images only, extracts and normalizes EXIF datetimes,
tags images (CLIP preferred, torchvision fallback), stores results in DuckDB
using UUID primary keys, and generates summary graphs and a CSV.

<<<<<<< HEAD
Features:
=======
Includes:
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
- Restrict to .jpg/.jpeg (case-insensitive)
- Sets PIL.Image.MAX_IMAGE_PIXELS = 933120000
- Normalizes EXIF datetime "YYYY:MM:DD HH:MM:SS" -> "YYYY-MM-DD HH:MM:SS"
- UUID text primary key (DuckDB has no autoincrement)
- CLIP processor uses channels-last numpy arrays with input_data_format
- Pandas explode/dropna fix to avoid assignment errors
<<<<<<< HEAD
- Command-line option --max-files (default: 1000) to limit number of files processed
"""

import argparse
=======
"""

>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
from pathlib import Path
from datetime import datetime
import uuid
import sys
<<<<<<< HEAD
import re

=======

# Increase PIL's allowed pixel count to avoid DecompressionBombError
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
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
<<<<<<< HEAD
except Exception as e:
    USE_CLIP = False
    print(f"[WARN] CLIP not available: {e}", file=sys.stderr)
=======
except Exception:
    USE_CLIP = False
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd

try:
    import torch
    from torchvision import transforms, models
<<<<<<< HEAD
except Exception as e:
    USE_TORCHVISION = False
    print(f"[WARN] Torchvision not available: {e}", file=sys.stderr)

=======
except Exception:
    USE_TORCHVISION = False

# ---------- Configuration ----------
ROOT_DIR = r"O:\Bilder\1-d7100"
DB_PATH = "photos_db.db"

# Only allow JPG/JPEG files (case-insensitive)
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
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

<<<<<<< HEAD

# ---------- Utility functions ----------
def norm_text(text):
    """Normalize text to lowercase and strip whitespace."""
    if not isinstance(text, str):
        return ""
    return text.lower().strip()


def normalize_exif_datetime(dt_str):
    """
    Convert EXIF datetime format "YYYY:MM:DD HH:MM:SS" to ISO format "YYYY-MM-DD HH:MM:SS".
    Returns None if format is invalid.
    """
    if not dt_str:
        return None
    try:
        # Replace colons with dashes in the date part only
        parts = dt_str.split()
        if len(parts) >= 1:
            date_part = parts[0].replace(":", "-")
            time_part = parts[1] if len(parts) > 1 else "00:00:00"
            return f"{date_part} {time_part}"
=======
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
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
    except Exception:
        pass
    return None

<<<<<<< HEAD

def season_from_iso(iso_datetime):
    """Determine season from ISO datetime string YYYY-MM-DD HH:MM:SS."""
    if not iso_datetime:
        return None
    try:
        dt = datetime.strptime(iso_datetime, "%Y-%m-%d %H:%M:%S")
        month = dt.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    except Exception:
        return None


def color_temperature_hint(image_path):
    """
    Estimate color temperature from image (simplified).
    Returns 'warm', 'cool', or 'neutral'.
    """
    try:
        img = PILImage.open(image_path).convert("RGB")
        stat = ImageStat.Stat(img)
        r_mean, g_mean, b_mean = stat.mean[0], stat.mean[1], stat.mean[2]
        
        if r_mean > b_mean + 10:
            return "warm"
        elif b_mean > r_mean + 10:
            return "cool"
        else:
            return "neutral"
    except Exception:
        return "unknown"


def extract_exif(image_path):
    """
    Extract EXIF metadata from JPG/JPEG image.
    Returns dictionary with normalized fields.
    """
    exif_data = {
        "file_path": str(image_path),
        "filename": image_path.name,
        "datetime_original": None,
        "make": None,
        "model": None,
        "lens_model": None,
        "focal_length": None,
        "iso": None,
        "aperture": None,
        "shutter_speed": None,
        "gps_lat": None,
        "gps_lon": None,
        "color_temperature": None,
        "image_width": None,
        "image_height": None,
    }
    
    try:
        # Get basic image info
        img = PILImage.open(image_path)
        exif_data["image_width"] = img.width
        exif_data["image_height"] = img.height
        exif_data["color_temperature"] = color_temperature_hint(image_path)
        
        # Extract EXIF with piexif
        try:
            exif_dict = piexif.load(str(image_path))
            
            # DateTime
            if "0th" in exif_dict:
                dt_bytes = exif_dict["0th"].get(piexif.ImageIFD.DateTime)
                if dt_bytes:
                    dt_str = dt_bytes.decode().strip("\x00")
                    exif_data["datetime_original"] = normalize_exif_datetime(dt_str)
            
            # Make, Model
            if "0th" in exif_dict:
                make_bytes = exif_dict["0th"].get(piexif.ImageIFD.Make)
                if make_bytes:
                    exif_data["make"] = norm_text(make_bytes.decode().strip("\x00"))
                
                model_bytes = exif_dict["0th"].get(piexif.ImageIFD.Model)
                if model_bytes:
                    exif_data["model"] = norm_text(model_bytes.decode().strip("\x00"))
            
            # Exif IFD
            if "Exif" in exif_dict:
                # LensModel
                lens_bytes = exif_dict["Exif"].get(piexif.ExifIFD.LensModel)
                if lens_bytes:
                    exif_data["lens_model"] = norm_text(lens_bytes.decode().strip("\x00"))
                
                # FocalLength
                fl = exif_dict["Exif"].get(piexif.ExifIFD.FocalLength)
                if fl:
                    try:
                        val = fl[0][0] / fl[0][1]
                        exif_data["focal_length"] = float(val)
                    except Exception:
                        pass
                
                # ISO
                iso = exif_dict["Exif"].get(piexif.ExifIFD.ISOSpeedRatings)
                if iso:
                    exif_data["iso"] = int(iso)
                
                # Aperture (F-Number)
                fn = exif_dict["Exif"].get(piexif.ExifIFD.FNumber)
                if fn:
                    try:
                        val = fn[0] / fn[1]
                        exif_data["aperture"] = float(val)
                    except Exception:
                        pass
                
                # ShutterSpeed (ExposureTime)
                ss = exif_dict["Exif"].get(piexif.ExifIFD.ExposureTime)
                if ss:
                    try:
                        val = ss[0] / ss[1]
                        exif_data["shutter_speed"] = float(val)
                    except Exception:
                        pass
            
            # GPS
            if "GPS" in exif_dict:
                gps_lat = exif_dict["GPS"].get(piexif.GPSIFD.GPSLatitude)
                gps_lon = exif_dict["GPS"].get(piexif.GPSIFD.GPSLongitude)
                if gps_lat:
                    try:
                        lat = gps_lat[0][0] / gps_lat[0][1] + \
                              gps_lat[1][0] / (gps_lat[1][1] * 60) + \
                              gps_lat[2][0] / (gps_lat[2][1] * 3600)
                        exif_data["gps_lat"] = float(lat)
                    except Exception:
                        pass
                if gps_lon:
                    try:
                        lon = gps_lon[0][0] / gps_lon[0][1] + \
                              gps_lon[1][0] / (gps_lon[1][1] * 60) + \
                              gps_lon[2][0] / (gps_lon[2][1] * 3600)
                        exif_data["gps_lon"] = float(lon)
                    except Exception:
                        pass
        
        except Exception as e:
            print(f"[WARN] Could not extract EXIF from {image_path}: {e}", file=sys.stderr)
    
    except Exception as e:
        print(f"[ERROR] Could not open {image_path}: {e}", file=sys.stderr)
    
    return exif_data


# ---------- Tagger class ----------
class Tagger:
    """Image tagger using CLIP (preferred) or torchvision ResNet50."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.use_clip = USE_CLIP
        
        if USE_CLIP:
            try:
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                print(f"[INFO] CLIP model loaded on {self.device}")
            except Exception as e:
                print(f"[WARN] Failed to load CLIP: {e}", file=sys.stderr)
                self.use_clip = False
        
        if not self.use_clip and USE_TORCHVISION:
            try:
                self.model = models.resnet50(pretrained=True).to(self.device)
                self.model.eval()
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                ])
                print(f"[INFO] ResNet50 model loaded on {self.device}")
            except Exception as e:
                print(f"[WARN] Failed to load ResNet50: {e}", file=sys.stderr)
    
    def predict_tags(self, image_path, topk=8):
        """Predict tags for image."""
        if not self.model:
            return []
        
        try:
            if self.use_clip:
                return self._predict_clip(image_path, topk)
            else:
                return self._predict_resnet(image_path, topk)
        except Exception as e:
            print(f"[WARN] Tagging failed for {image_path}: {e}", file=sys.stderr)
            return []
    
    def _predict_clip(self, image_path, topk=8):
        """CLIP-based tagging."""
        try:
            img = PILImage.open(image_path).convert("RGB")
            
            # Common scene/object labels
            labels = [
                "photo of a person", "photo of people", "photo of a child",
                "photo of a bird", "photo of birds", "photo of a dog", "photo of a cat",
                "photo of a tree", "photo of trees", "photo of a forest",
                "photo of a mountain", "photo of mountains", "photo of a landscape",
                "photo of water", "photo of a lake", "photo of a river", "photo of the ocean",
                "photo of the sky", "photo of clouds", "photo of a sunset", "photo of a sunrise",
                "photo of snow", "photo of flowers", "photo of a flower",
                "photo of a building", "photo of buildings", "photo of a street",
                "photo of a car", "photo of cars", "photo of a house",
            ]
            
            inputs = self.processor(text=labels, images=img, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
            
            probs = logits_per_image.softmax(dim=1)[0]
            top_indices = torch.topk(probs, min(topk, len(labels))).indices
            tags = [labels[i].replace("photo of ", "").replace("a ", "").strip() for i in top_indices]
            return tags
        
        except Exception as e:
            print(f"[WARN] CLIP prediction failed: {e}", file=sys.stderr)
            return []
    
    def _predict_resnet(self, image_path, topk=8):
        """ResNet50-based tagging (fallback)."""
        try:
            img = PILImage.open(image_path).convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
            
            # Simple top-k labels (ResNet1000 classes)
            probs = torch.softmax(outputs, dim=1)[0]
            top_probs, top_indices = torch.topk(probs, topk)
            tags = [f"class_{i}" for i in top_indices.tolist()]
            return tags
        
        except Exception as e:
            print(f"[WARN] ResNet prediction failed: {e}", file=sys.stderr)
            return []


# ---------- Categorization ----------
def map_tags_to_categories(tags):
    """Map tag list to coarse categories."""
    categories = []
    for tag in tags:
        tag_lower = norm_text(tag)
        for category, keywords in COARSE_CATEGORIES.items():
            if any(kw in tag_lower for kw in keywords):
                if category not in categories:
                    categories.append(category)
                break
    return categories


# ---------- Database operations ----------
def init_db(conn):
    """Initialize DuckDB schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            filename TEXT,
            datetime_original TEXT,
            make TEXT,
            model TEXT,
            lens_model TEXT,
            focal_length DOUBLE,
            iso INTEGER,
            aperture DOUBLE,
            shutter_speed DOUBLE,
            gps_lat DOUBLE,
            gps_lon DOUBLE,
            color_temperature TEXT,
            image_width INTEGER,
            image_height INTEGER,
            tags TEXT[],
            categories TEXT[],
            season TEXT,
            year INTEGER,
            month INTEGER
        )
    """)
    print("[INFO] Database schema initialized")


def insert_record(conn, record):
    """Insert a photo record into DuckDB."""
    record_id = str(uuid.uuid4())
    
    conn.execute("""
        INSERT INTO photos VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, [
        record_id,
        record.get("file_path"),
        record.get("filename"),
        record.get("datetime_original"),
        record.get("make"),
        record.get("model"),
        record.get("lens_model"),
        record.get("focal_length"),
        record.get("iso"),
        record.get("aperture"),
        record.get("shutter_speed"),
        record.get("gps_lat"),
        record.get("gps_lon"),
        record.get("color_temperature"),
        record.get("image_width"),
        record.get("image_height"),
        record.get("tags"),
        record.get("categories"),
        record.get("season"),
        record.get("year"),
        record.get("month")
    ])


# ---------- Reporting ----------
def generate_reports(conn):
    """Generate summary statistics and graphs."""
    try:
        df = conn.execute("SELECT * FROM photos").df()
        
        if len(df) == 0:
            print("[WARN] No photos in database; skipping reports")
            return
        
        print(f"[INFO] Generating reports for {len(df)} photos...")
        
        # CSV export
        csv_path = "photos_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV exported to {csv_path}")
        
        # Year distribution
        if "year" in df.columns and df["year"].notna().sum() > 0:
            plt.figure(figsize=(10, 4))
            df[df["year"].notna()]["year"].value_counts().sort_index().plot(kind="bar")
            plt.title("Photos by Year")
            plt.xlabel("Year")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("report_year_distribution.png", dpi=100)
            plt.close()
            print("[INFO] Year distribution graph saved")
        
        # Season distribution
        if "season" in df.columns and df["season"].notna().sum() > 0:
            plt.figure(figsize=(8, 4))
            df[df["season"].notna()]["season"].value_counts().plot(kind="bar", color="steelblue")
            plt.title("Photos by Season")
            plt.xlabel("Season")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig("report_season_distribution.png", dpi=100)
            plt.close()
            print("[INFO] Season distribution graph saved")
        
        # Category distribution (exploded)
        if "categories" in df.columns:
            cat_series = df["categories"].explode()
            if len(cat_series) > 0:
                plt.figure(figsize=(10, 5))
                cat_series.value_counts().plot(kind="barh", color="coral")
                plt.title("Photos by Category")
                plt.xlabel("Count")
                plt.tight_layout()
                plt.savefig("report_category_distribution.png", dpi=100)
                plt.close()
                print("[INFO] Category distribution graph saved")
        
        # Make distribution
        if "make" in df.columns and df["make"].notna().sum() > 0:
            plt.figure(figsize=(8, 4))
            df[df["make"].notna()]["make"].value_counts().head(10).plot(kind="barh", color="green")
            plt.title("Photos by Camera Make (Top 10)")
            plt.xlabel("Count")
            plt.tight_layout()
            plt.savefig("report_make_distribution.png", dpi=100)
            plt.close()
            print("[INFO] Make distribution graph saved")
        
        # Summary stats
        stats = {
            "total_photos": len(df),
            "photos_with_datetime": df["datetime_original"].notna().sum(),
            "photos_with_gps": (df["gps_lat"].notna() & df["gps_lon"].notna()).sum(),
            "avg_width": df["image_width"].mean(),
            "avg_height": df["image_height"].mean(),
        }
        
        print("\n=== SUMMARY STATISTICS ===")
        for k, v in stats.items():
            print(f"{k}: {v}")
        print("=" * 40 + "\n")
    
    except Exception as e:
        print(f"[ERROR] Report generation failed: {e}", file=sys.stderr)


# ---------- Main pipeline ----------
def scan_and_analyze(root_dir, db_path, max_files):
    """
    Scan directory for JPG/JPEG images, extract EXIF, tag with ML, store in DuckDB.
    
    Args:
        root_dir: Root directory to scan
        db_path: Path to DuckDB database file
        max_files: Maximum number of files to process (default 1000)
    """
    p = Path(root_dir)
    if not p.exists():
        print(f"[ERROR] Root directory {root_dir} not found.", file=sys.stderr)
        sys.exit(1)
    
    # Validate max_files
    if max_files <= 0:
        print(f"[ERROR] --max-files must be positive (got {max_files})", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Starting scan with --max-files limit: {max_files}")
    
=======
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
            # fallback omitted for brevity; piexif usually covers common EXIF fields
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
            img = PILImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {image_path}: {e}", file=sys.stderr)
            return tags

        # CLIP branch: channels-last numpy array and explicit input_data_format
        if self.use_clip:
            try:
                keyword_texts = ["a photo of a " + k for k in sum(COARSE_CATEGORIES.values(), [])]
                img_arr = np.asarray(img)  # (H, W, C)
                inputs = self.clip_processor(
                    text=keyword_texts,
                    images=[img_arr],
                    return_tensors="pt",
                    padding=True,
                    input_data_format="channels_last"
                )
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.clip_model(**inputs)
                    image_emb = out.image_embeds
                    text_emb = out.text_embeds
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    sims = (100.0 * image_emb @ text_emb.T).softmax(dim=-1)
                    topk_idx = sims[0].argsort(descending=True)[:topk].cpu().numpy().tolist()
                    tags = [keyword_texts[i].replace("a photo of a ", "") for i in topk_idx]
            except Exception as e:
                print(f"[INFO] CLIP predict failed, falling back: {e}", file=sys.stderr)
                tags = []

        # torchvision fallback
        if not tags and self.use_torchvision:
            try:
                x = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(x)
                    probs = torch.nn.functional.softmax(out, dim=1)[0]
                    topk_idx = torch.topk(probs, topk).indices.cpu().numpy().tolist()
                    tags = [self.imagenet_labels[i] if i < len(self.imagenet_labels) else f"label{i}" for i in topk_idx]
            except Exception as e:
                print(f"[INFO] torchvision predict failed: {e}", file=sys.stderr)
                tags = []

        tags = [norm_text(str(t)) for t in tags if t]
        ct = color_temperature_hint(img)
        if ct:
            tags.append(ct)
        return list(dict.fromkeys(tags))

def map_tags_to_categories(tags):
    cats = set()
    for cat, keywords in COARSE_CATEGORIES.items():
        for kw in keywords:
            for t in tags:
                if kw in t:
                    cats.add(cat)
    return sorted(list(cats))

# ---------- Database ----------
def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS photos (
        id TEXT PRIMARY KEY,
        file_path TEXT,
        file_name TEXT,
        width INTEGER,
        height INTEGER,
        datetime_original TEXT,
        year INTEGER,
        month INTEGER,
        season TEXT,
        camera_make TEXT,
        camera_model TEXT,
        exposure_time TEXT,
        f_number TEXT,
        iso INTEGER,
        focal_length TEXT,
        tags TEXT,
        categories TEXT
    );
    """)
    conn.commit()

def insert_record(conn, rec):
    rid = rec.get("id") or uuid.uuid4().hex
    conn.execute("""
    INSERT INTO photos (
        id, file_path, file_name, width, height, datetime_original, year, month, season,
        camera_make, camera_model, exposure_time, f_number, iso, focal_length, tags, categories
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rid,
        rec.get("file_path"),
        rec.get("file_name"),
        rec.get("width"),
        rec.get("height"),
        rec.get("datetime_original"),
        rec.get("year"),
        rec.get("month"),
        rec.get("season"),
        rec.get("camera_make"),
        rec.get("camera_model"),
        str(rec.get("exposure_time")) if rec.get("exposure_time") is not None else None,
        str(rec.get("f_number")) if rec.get("f_number") is not None else None,
        rec.get("iso"),
        str(rec.get("focal_length")) if rec.get("focal_length") is not None else None,
        ",".join(rec.get("tags") or []),
        ",".join(rec.get("categories") or [])
    ))
    conn.commit()
    return rid

# ---------- Reporting ----------
def generate_reports(conn):
    df = conn.execute("SELECT * FROM photos").fetchdf()
    if df.empty:
        print("[INFO] No photos found in DB to report.")
        return

    # Build categories_list, drop rows with empty categories before using explode
    df["categories_list"] = df["categories"].fillna("").apply(lambda s: s.split(",") if s else [])
    df_expl = df.explode("categories_list")
    df_expl["categories_list"] = df_expl["categories_list"].replace("", pd.NA)
    df_expl = df_expl.dropna(subset=["categories_list"])

    sns.set(style="whitegrid")

    # Photo type counts
    plt.figure(figsize=(10,6))
    type_counts = df_expl["categories_list"].value_counts()
    sns.barplot(x=type_counts.values, y=type_counts.index, palette="viridis")
    plt.title("Photo type counts")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("photo_type_counts.png", dpi=150)
    plt.close()

    # Photos per year
    plt.figure(figsize=(10,6))
    year_counts = df["year"].value_counts().sort_index()
    sns.barplot(x=year_counts.index.astype(str), y=year_counts.values, palette="magma")
    plt.title("Photos per year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("photos_per_year.png", dpi=150)
    plt.close()

    # ISO distribution
    df_iso = df.dropna(subset=["iso"])
    if not df_iso.empty:
        plt.figure(figsize=(10,5))
        sns.histplot(df_iso["iso"].astype(int), bins=30, kde=False, color="steelblue")
        plt.title("ISO distribution")
        plt.xlabel("ISO")
        plt.tight_layout()
        plt.savefig("iso_distribution.png", dpi=150)
        plt.close()

    # Aperture distribution
    def parse_rat(r):
        if pd.isna(r):
            return None
        try:
            s = str(r)
            if "/" in s:
                a,b = s.split("/")
                return float(a)/float(b)
            return float(s)
        except Exception:
            return None

    df["f_number_parsed"] = df["f_number"].apply(parse_rat)
    df_fn = df.dropna(subset=["f_number_parsed"])
    if not df_fn.empty:
        plt.figure(figsize=(8,4))
        sns.countplot(x=df_fn["f_number_parsed"].round(1).astype(str),
                      palette="coolwarm",
                      order=sorted(df_fn["f_number_parsed"].unique()))
        plt.title("Aperture (F-number) distribution")
        plt.xlabel("F-number")
        plt.tight_layout()
        plt.savefig("aperture_distribution.png", dpi=150)
        plt.close()

    # Top camera models
    top_models = df["camera_model"].fillna("Unknown").value_counts().head(10)
    plt.figure(figsize=(10,5))
    sns.barplot(y=top_models.index, x=top_models.values, palette="cubehelix")
    plt.title("Top camera models")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig("top_camera_models.png", dpi=150)
    plt.close()

    df.to_csv("photos_summary.csv", index=False)
    print("[INFO] Generated graphs and photos_summary.csv in current directory.")

# ---------- Main pipeline ----------
def scan_and_analyze(root_dir, db_path):
    p = Path(root_dir)
    if not p.exists():
        print(f"[ERROR] Root directory {root_dir} not found.", file=sys.stderr)
        return
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
    conn = duckdb.connect(db_path)
    init_db(conn)
    tagger = Tagger()

<<<<<<< HEAD
    files = [fp for fp in p.rglob("*") if fp.suffix.lower() in IMAGE_EXTS]
    files = sorted(set(files))[:max_files]
    print(f"[INFO] Found {len(files)} JPG/JPEG images (processing max {max_files}). Processing...")

    for fp in tqdm(files, desc="Processing images"):
        try:
            exif = extract_exif(fp)
            tags = tagger.predict_tags(fp, topk=8)
            categories = map_tags_to_categories(tags)
            season = season_from_iso(exif.get("datetime_original"))
            if season and season not in categories:
                categories.append(season)
            
            year = None
            month = None
            dto = exif.get("datetime_original")
            if dto:
                try:
                    dt = datetime.strptime(dto, "%Y-%m-%d %H:%M:%S")
                    year = dt.year
                    month = dt.month
                except Exception:
                    pass
            
            rec = dict(exif)
            rec.update({
                "tags": tags,
                "categories": categories,
                "season": season,
                "year": year,
                "month": month
            })
            insert_record(conn, rec)
        
        except Exception as e:
            print(f"[ERROR] Failed to process {fp}: {e}", file=sys.stderr)
            continue

    conn.commit()
=======
    # gather JPG/JPEG image files: suffix lowercased to match .jpg/.jpeg
    files = [fp for fp in p.rglob("*") if fp.suffix.lower() in IMAGE_EXTS]
    files = sorted(set(files))
    print(f"[INFO] Found {len(files)} JPG/JPEG images. Processing...")

    for fp in tqdm(files):
        exif = extract_exif(fp)
        tags = tagger.predict_tags(fp, topk=8)
        categories = map_tags_to_categories(tags)
        season = season_from_iso(exif.get("datetime_original"))
        if season and season not in categories:
            categories.append(season)
        year = None
        month = None
        dto = exif.get("datetime_original")
        if dto:
            try:
                dt = datetime.strptime(dto, "%Y-%m-%d %H:%M:%S")
                year = dt.year
                month = dt.month
            except Exception:
                year = None
                month = None
        rec = dict(exif)
        rec.update({"tags": tags, "categories": categories, "season": season, "year": year, "month": month})
        insert_record(conn, rec)

>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
    print("[INFO] Processing complete. Generating reports...")
    generate_reports(conn)
    conn.close()
    print(f"[INFO] Done. Database saved to {db_path}.")

<<<<<<< HEAD

# ---------- Entry point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze JPG/JPEG photos and store metadata in DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process up to 1000 images (default)
  python photo_analyze_jpg_only.py --root-dir ./photos

  # Process up to 500 images
  python photo_analyze_jpg_only.py --root-dir ./photos --max-files 500

  # Process up to 5000 images
  python photo_analyze_jpg_only.py --root-dir ./photos --max-files 5000 --db-path custom.db
        """
    )
    
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory to scan for JPG/JPEG images"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="photos_db.db",
        help="Path to DuckDB database file (default: photos_db.db)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Maximum number of image files to process (default: 1000)"
    )
    
    args = parser.parse_args()

    scan_and_analyze(args.root_dir, args.db_path, args.max_files)
=======
# Keep init_db near bottom to avoid forward reference confusion in some environments
def init_db(conn):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS photos (
        id TEXT PRIMARY KEY,
        file_path TEXT,
        file_name TEXT,
        width INTEGER,
        height INTEGER,
        datetime_original TEXT,
        year INTEGER,
        month INTEGER,
        season TEXT,
        camera_make TEXT,
        camera_model TEXT,
        exposure_time TEXT,
        f_number TEXT,
        iso INTEGER,
        focal_length TEXT,
        tags TEXT,
        categories TEXT
    );
    """)
    conn.commit()

if __name__ == "__main__":
    scan_and_analyze(ROOT_DIR, DB_PATH)
>>>>>>> 2ccbf681e51eec9285bf66c5efc4572a0f135acd
