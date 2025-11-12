#!/usr/bin/env python3
"""
photo_analyze_download_fixed.py

Scans ROOT_DIR for images, extracts and normalizes EXIF datetimes,
tags images (CLIP preferred, torchvision fallback), stores results in DuckDB
using UUID primary keys, and generates summary graphs and a CSV.

Key fixes:
- Sets PIL.Image.MAX_IMAGE_PIXELS = 933120000
- Avoids pandas shape-mismatch by dropping empty categories before assignment
"""

from pathlib import Path
from datetime import datetime
import uuid
import sys
import os

from collections import defaultdict

# Increase PIL's allowed pixel count to avoid DecompressionBombError on very large images
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
    from transformers.image_utils import ChannelDimension
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".heic", ".heif", ".bmp", ".gif"}

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
    """
    Convert EXIF datetime like '2011:05:29 23:26:46' to 'YYYY-MM-DD HH:MM:SS'.
    Return None if parsing fails.
    """
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
            try:
                pil_exif = img._getexif()
                if pil_exif:
                    tag_map = {v:k for k,v in ExifTags.TAGS.items()}
                    dto_tag = tag_map.get("DateTimeOriginal") or 36867
                    make_tag = tag_map.get("Make") or 271
                    model_tag = tag_map.get("Model") or 272
                    iso_tag = tag_map.get("ISOSpeedRatings") or 34855
                    exp_tag = tag_map.get("ExposureTime") or 33434
                    fn_tag = tag_map.get("FNumber") or 33437
                    fl_tag = tag_map.get("FocalLength") or 37386
                    if dto_tag in pil_exif:
                        data["datetime_original"] = normalize_exif_datetime(pil_exif[dto_tag])
                    if make_tag in pil_exif:
                        data["camera_make"] = pil_exif[make_tag]
                    if model_tag in pil_exif:
                        data["camera_model"] = pil_exif[model_tag]
                    if iso_tag in pil_exif:
                        try:
                            data["iso"] = int(pil_exif[iso_tag])
                        except:
                            data["iso"] = None
                    if exp_tag in pil_exif:
                        data["exposure_time"] = pil_exif[exp_tag]
                    if fn_tag in pil_exif:
                        data["f_number"] = pil_exif[fn_tag]
                    if fl_tag in pil_exif:
                        data["focal_length"] = pil_exif[fl_tag]
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
            except Exception as e:
                print(f"[INFO] CLIP unavailable: {e}", file=sys.stderr)
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
            except Exception as e:
                print(f"[INFO] torchvision model unavailable: {e}", file=sys.stderr)
                self.use_torchvision = False

    def predict_tags(self, image_path: Path, topk=8):
        tags = []
        try:
            img = PILImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Cannot open {image_path}: {e}", file=sys.stderr)
            return tags

        # CLIP branch: ensure channels-last numpy array and explicit input_data_format
        if self.use_clip:
            try:
                keyword_texts = ["a photo of a " + k for k in sum(COARSE_CATEGORIES.values(), [])]
                img_arr = np.asarray(img)  # shape (H, W, C) channels-last
                inputs = self.clip_processor(
                    text=keyword_texts,
                    images=[img_arr],
                    return_tensors="pt",
                    padding=True,
                    input_data_format="channels_last"
                )
                # move tensors to device
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
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

    # Build categories_list, drop rows with empty categories before using explode to avoid shape-mismatch
    df["categories_list"] = df["categories"].fillna("").apply(lambda s: s.split(",") if s else [])
    # explode
    df_expl = df.explode("categories_list")
    # now drop rows where categories_list is empty string or NA
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
        sns.countplot(x=df_fn["f_number_parsed"].round(1).astype(str), palette="coolwarm",
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
    conn = duckdb.connect(db_path)
    init_db(conn)
    tagger = Tagger()
    # gather image files
    files = []
    for ext in IMAGE_EXTS:
        files.extend(p.rglob(f"*{ext}"))
    files = sorted(set(files))
    print(f"[INFO] Found {len(files)} images. Processing...")
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
        rec.update({
            "tags": tags,
            "categories": categories,
            "season": season,
            "year": year,
            "month": month
        })
        insert_record(conn, rec)
    print("[INFO] Processing complete. Generating reports...")
    generate_reports(conn)
    conn.close()
    print(f"[INFO] Done. Database saved to {db_path}.")

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
