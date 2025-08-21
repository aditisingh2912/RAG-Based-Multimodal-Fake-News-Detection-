import os
import io
import time
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss


# =========================
# Config (edit if you want)
# =========================
DEFAULT_KB_DIR = "knowledge_base"  # each sample in its own subfolder
IMAGE_FILE_NAME = "Image.jpg"      # required
CAPTION_FILE_NAME = "caption.txt"  # required
GT_FILE_NAME = "GT.txt"            # optional (e.g., "Real"/"Fake" or any tag)


# =========================
# Utilities
# =========================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms

def softmax_1d(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (scores / max(temperature, 1e-6)) - (scores.max() if scores.size else 0.0)
    e = np.exp(z)
    s = e.sum() + 1e-12
    return e / s


@dataclass
class KBItem:
    caption: str
    image_path: str
    gt: Optional[str] = None


# =========================
# Model load (cached)
# =========================
@st.cache_resource(show_spinner=True)
def load_clip():
    model_name = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


# =========================
# Build Knowledge Base
# =========================
def _iter_kb_samples(kb_dir: str) -> List[KBItem]:
    items: List[KBItem] = []
    if not os.path.isdir(kb_dir):
        return items

    for root, dirs, files in os.walk(kb_dir):
        if IMAGE_FILE_NAME in files and CAPTION_FILE_NAME in files:
            img_path = os.path.join(root, IMAGE_FILE_NAME)
            cap_path = os.path.join(root, CAPTION_FILE_NAME)
            gt_path = os.path.join(root, GT_FILE_NAME) if GT_FILE_NAME in files else None

            try:
                with open(cap_path, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
                gt = None
                if gt_path and os.path.exists(gt_path):
                    with open(gt_path, "r", encoding="utf-8") as f:
                        gt = f.read().strip()
                items.append(KBItem(caption=caption, image_path=img_path, gt=gt))
            except Exception:
                # Skip corrupt entries silently
                pass
    return items


@st.cache_resource(show_spinner=True)
def build_faiss_index(kb_dir: str, device: str):
    """
    Build the FAISS index over fused (text+image) CLIP embeddings for every KB sample.
    Returns:
        index: FAISS IndexFlatIP over unit-norm vectors (cosine similarity)
        kb_items: list[KBItem]
        dim: embedding dimension
    """
    model, processor, _device = load_clip()
    kb_items = _iter_kb_samples(kb_dir)

    if not kb_items:
        # empty KB fallback to avoid FAISS crash
        dim = model.config.projection_dim
        index = faiss.IndexFlatIP(dim)
        return index, [], dim

    embs = []
    for item in kb_items:
        try:
            img = Image.open(item.image_path).convert("RGB")
            inputs = processor(text=[item.caption], images=img,
                               return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                # Better practice: use modality-specific encoders
                img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
                txt_feat = model.get_text_features(input_ids=inputs["input_ids"],
                                                   attention_mask=inputs["attention_mask"])
            img_feat = img_feat.cpu().numpy().astype("float32")
            txt_feat = txt_feat.cpu().numpy().astype("float32")

            # Fuse text+image (average) then re-normalize
            fused = l2_normalize((img_feat + txt_feat) / 2.0)
            embs.append(fused)
        except Exception:
            # Skip corrupt sample but continue
            pass

    if not embs:
        dim = model.config.projection_dim
        index = faiss.IndexFlatIP(dim)
        return index, [], dim

    mat = l2_normalize(np.vstack(embs).astype("float32"))
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via dot on unit vectors
    index.add(mat)
    return index, kb_items, dim


# =========================
# Query / Scoring
# =========================
def encode_pair_to_fused_emb(text: str, image: Image.Image, device: str) -> Tuple[np.ndarray, float, float]:
    """
    Returns fused embedding AND the direct image↔text cosine similarity
    (the latter serves as a consistency signal independent of KB).
    """
    model, processor, _ = load_clip()
    t0 = time.time()
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = model.get_text_features(input_ids=inputs["input_ids"],
                                           attention_mask=inputs["attention_mask"])
        # Normalize to compute cosine
        img_unit = torch.nn.functional.normalize(img_feat, p=2, dim=1)
        txt_unit = torch.nn.functional.normalize(txt_feat, p=2, dim=1)
        pair_cosine = torch.sum(img_unit * txt_unit, dim=1).item()

    fused = (img_feat.cpu().numpy() + txt_feat.cpu().numpy()) / 2.0
    fused = l2_normalize(fused.astype("float32"))
    elapsed = time.time() - t0
    return fused, float(pair_cosine), float(elapsed)


def predict_real_fake(
    user_text: str,
    user_image: Image.Image,
    index: faiss.IndexFlatIP,
    kb_items: List[KBItem],
    alpha: float = 0.6,          # weight for direct (image,text) consistency
    temperature: float = 0.07,   # softmax temp for KB evidence
    top_k: int = 5,
    threshold: float = 0.55      # final decision threshold on blended score
):
    """
    1) Encode user text+image -> fused embedding + direct cosine
    2) Search KB for top-k evidence using FAISS cosine similarity
    3) Convert KB sims to probability via softmax
    4) Blend score = alpha*pair_cosine + (1-alpha)*p_max
    5) Decision = Real if blend >= threshold else Fake
    """
    fused, pair_cos, enc_time = encode_pair_to_fused_emb(user_text, user_image, device)

    # Retrieval
    k = min(top_k, index.ntotal) if index.ntotal > 0 else 0
    if k > 0:
        sims, ids = index.search(fused, k)
        sims = sims[0]          # cosine similarities
        ids = ids[0].tolist()
        kb_probs = softmax_1d(sims, temperature=temperature)
        p_max = float(kb_probs[0]) if kb_probs.size else 0.0
    else:
        sims, ids = np.array([]), []
        kb_probs = np.array([])
        p_max = 0.0

    # Blend the direct consistency with KB evidence
    # map pair_cos from [-1,1] -> [0,1] to combine fairly
    pair_prob = (pair_cos + 1.0) / 2.0
    blended = alpha * pair_prob + (1 - alpha) * p_max

    label = "Real" if blended >= threshold else "Fake"

    # Prepare evidence list
    evidence = []
    for rank, idx in enumerate(ids):
        item = kb_items[idx]
        evidence.append({
            "rank": rank + 1,
            "caption": item.caption,
            "image_path": item.image_path,
            "gt": item.gt,
            "cosine": float(sims[rank]),
            "prob": float(kb_probs[rank]) if kb_probs.size else None
        })

    return {
        "label": label,
        "pair_cosine": pair_cos,            # direct image↔text consistency (cosine)
        "kb_top_prob": p_max,               # prob mass on top KB evidence
        "confidence": blended,              # final blended score in [0,1]
        "encode_time_sec": enc_time,
        "evidence": evidence
    }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Fake Image–Text Detector (CLIP L/14 + FAISS)", layout="centered", page_icon="📰")
st.title("📰 Fake Image–Text Detector")
st.caption("CLIP ViT-L/14 + FAISS · Evidence-backed predictions · Streamlit-Cloud ready")

model, processor, device = load_clip()

with st.sidebar:
    st.header("⚙️ Settings")
    kb_dir = st.text_input("Knowledge base folder", value=DEFAULT_KB_DIR)
    alpha = st.slider("Blend weight α (direct consistency vs KB)", 0.0, 1.0, 0.6, 0.05)
    threshold = st.slider("Decision threshold (higher = stricter 'Real')", 0.3, 0.9, 0.55, 0.01)
    temperature = st.slider("KB softmax temperature", 0.01, 1.0, 0.07, 0.01)
    top_k = st.slider("Top-K evidence to retrieve", 1, 10, 5, 1)
    st.caption(f"Device: **{device.upper()}**")

# Build / rebuild KB index when kb_dir changes
if "kb_dir_cached" not in st.session_state or st.session_state["kb_dir_cached"] != kb_dir:
    with st.spinner("Building FAISS index from knowledge base..."):
        index, kb_items, dim = build_faiss_index(kb_dir, device)
        st.session_state["faiss_index"] = index
        st.session_state["kb_items"] = kb_items
        st.session_state["kb_dir_cached"] = kb_dir

index: faiss.IndexFlatIP = st.session_state.get("faiss_index", None)
kb_items: List[KBItem] = st.session_state.get("kb_items", [])

if not kb_items:
    st.warning(
        f"No KB samples found in **{kb_dir}**.\n\n"
        f"Expected structure per sample folder:\n"
        f"- `{IMAGE_FILE_NAME}` (image)\n"
        f"- `{CAPTION_FILE_NAME}` (text caption)\n"
        f"- `{GT_FILE_NAME}` (optional ground-truth tag)\n"
    )

st.markdown("---")
with st.form("predict_form"):
    text_input = st.text_area("📝 Enter the description for the uploaded image:", height=100)
    img_file = st.file_uploader("🖼️ Upload an image (JPG/PNG):", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    if not text_input or not img_file:
        st.warning("Please provide both text and image.")
    else:
        try:
            img = Image.open(io.BytesIO(img_file.read())).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Running CLIP + FAISS..."):
                t0 = time.time()
                result = predict_real_fake(
                    user_text=text_input,
                    user_image=img,
                    index=index,
                    kb_items=kb_items,
                    alpha=alpha,
                    temperature=temperature,
                    top_k=top_k,
                    threshold=threshold
                )
                total_time = time.time() - t0

            # Display main metrics
            st.subheader("Result")
            st.metric(
                "Prediction",
                result["label"],
                delta=f"Confidence: {result['confidence']*100:.2f}%"
            )
            st.write(f"⏱ Inference time (total): **{total_time:.2f}s**  "
                     f"(encode: {result['encode_time_sec']:.2f}s)")
            st.write(f"🔗 Image↔Text cosine: **{result['pair_cosine']:.3f}**  ·  "
                     f"📚 KB top probability: **{result['kb_top_prob']:.3f}**")

            # Evidence
            st.markdown("### 🔍 Evidence retrieved from knowledge base")
            if not result["evidence"]:
                st.info("No evidence (empty KB).")
            else:
                for ev in result["evidence"]:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        try:
                            st.image(ev["image_path"], use_container_width=True)
                        except Exception:
                            st.write("Image not found.")
                    with cols[1]:
                        st.markdown(
                            f"**#{ev['rank']}** · Cosine: `{ev['cosine']:.3f}`"
                            + (f" · Prob: `{ev['prob']:.3f}`" if ev["prob"] is not None else "")
                        )
                        st.markdown(f"**Caption:** {ev['caption']}")
                        if ev["gt"]:
                            st.caption(f"GT: {ev['gt']}")

        except Exception as e:
            st.error(f"Failed to process input: {e}")

st.markdown("---")
st.caption("Tip: Add more domain-specific examples to the KB to improve retrieval evidence and robustness.")

