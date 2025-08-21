# Training_pipeline/app.py
import os
import time
import faiss
import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------
# Paths & device
# -----------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # one level up from Training_pipeline/
KB_DIR = os.path.join(ROOT_DIR, "knowledge_base")
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-large-patch14"

# -----------------------
# Helpers
# -----------------------
def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / norms

def softmax_1d(scores: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    s = scores / max(temperature, 1e-8)
    s = s - s.max()
    e = np.exp(s)
    return e / (e.sum() + 1e-12)

# -----------------------
# Load CLIP (cached)
# -----------------------
@st.cache_resource(show_spinner=True)
def load_clip():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor

model, processor = load_clip()

# -----------------------
# Build multimodal KB (fused embeddings) and FAISS index
# -----------------------
@st.cache_resource(show_spinner=True)
def build_kb_index(kb_dir: str):
    items = []   # list of dicts: {caption, image_path, gt}
    fused_embs = []

    if not os.path.isdir(kb_dir):
        return None, []

    for sample_name in sorted(os.listdir(kb_dir)):
        sample_path = os.path.join(kb_dir, sample_name)
        if not os.path.isdir(sample_path):
            continue

        img_path = os.path.join(sample_path, "Image.jpg")
        cap_path = os.path.join(sample_path, "caption.txt")
        gt_path = os.path.join(sample_path, "GT.txt")

        if not (os.path.exists(img_path) and os.path.exists(cap_path)):
            continue

        # read caption
        try:
            with open(cap_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        except Exception:
            continue

        # load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # encode text
        text_inputs = processor(text=[caption], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_feat = model.get_text_features(**text_inputs)  # (1, d)
        text_np = text_feat.cpu().numpy().astype("float32")

        # encode image
        image_inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_feat = model.get_image_features(**image_inputs)  # (1, d)
        img_np = img_feat.cpu().numpy().astype("float32")

        # normalize and fuse (average), then renormalize
        text_unit = l2_normalize_rows(text_np)
        img_unit = l2_normalize_rows(img_np)
        fused = (text_unit + img_unit) / 2.0
        fused = l2_normalize_rows(fused)

        fused_embs.append(fused[0])
        items.append({"name": sample_name, "caption": caption, "image_path": img_path, "gt": (open(gt_path).read().strip() if os.path.exists(gt_path) else None)})

    if len(fused_embs) == 0:
        return None, []

    mat = np.vstack(fused_embs).astype("float32")  # (N, d)
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)   # use inner product on normalized vectors => cosine
    index.add(mat)
    return index, items

index, kb_items = build_kb_index(KB_DIR)

# -----------------------
# Predict function
# -----------------------
def predict(text, image):
    start_time = time.time()

    # Encode text
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_emb = model.get_text_features(**text_inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)   # normalize
    text_emb = text_emb.cpu().numpy()

    # Encode image
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**image_inputs)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)   # normalize
    image_emb = image_emb.cpu().numpy()

    # Cosine similarity
    sim_text_img = cosine_similarity(text_emb, image_emb)[0][0]

    # Retrieve top-3 evidence (based on caption similarity to text)
    D, I = index.search(text_emb.astype("float32"), k=3)
    evidence = [kb_meta[i] for i in I[0]]

    # Dynamic threshold for Real vs Fake
    label = "Real" if sim_text_img >= 0.28 else "Fake"

    end_time = time.time()
    infer_time = round(end_time - start_time, 3)

    return label, round(sim_text_img, 3), infer_time, evidence

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fake Image-Text Detector (CLIP+FAISS)", layout="centered")
st.title("🕵️ Fake Image–Text Detector (CLIP ViT-L/14 + FAISS)")
st.markdown("Enter an image and a caption to check whether the caption correctly describes the image. The model returns a confidence score plus top-k evidence from the knowledge base.")

# sidebar controls for quick tuning
with st.sidebar:
    st.header("Settings / Tuning")
    alpha = st.slider("Blend weight α (pair vs KB)", 0.0, 1.0, 0.6, 0.05)
    threshold = st.slider("Decision threshold (0-1)", 0.2, 0.9, 0.55, 0.01)
    temperature = st.slider("KB softmax temperature", 0.01, 1.0, 0.07, 0.01)
    top_k = st.slider("Top-K evidence", 1, 10, 5, 1)
    st.caption("If evidence looks unrelated, add more high-quality samples in knowledge_base/ and ensure caption.txt matches the image.")

st.markdown("---")
with st.form("form"):
    user_text = st.text_area("Enter caption / claim to verify:", height=100)
    uploaded = st.file_uploader("Upload an image (jpg/png):", type=["jpg", "jpeg", "png"])
    run = st.form_submit_button("Check")

if run:
    if (not user_text) or (not uploaded):
        st.warning("Please provide both a caption and an image.")
    else:
        try:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)

            with st.spinner("Running CLIP & FAISS..."):
                result = predict_pair(user_text, img, alpha=alpha, temperature=temperature, top_k=top_k, threshold=threshold)

            # main results
            st.subheader("Result")
            st.metric("Prediction", result["label"], f"{result['confidence']*100:.2f}%")
            st.write(f"Pair cosine: **{result['pair_cos']:.3f}** → mapped prob **{result['pair_prob']:.3f}**")
            st.write(f"KB top cosine: **{result['kb_top_sim']:.3f}**, KB top prob: **{result['kb_top_prob']:.3f}**")
            st.write(f"Total inference time: **{result['infer_time']:.2f}s**")

            st.markdown("### 🔎 Retrieved evidence (top-k)")
            if len(result["evidence"]) == 0:
                st.info("No knowledge-base evidence found (empty or path issue).")
            else:
                for ev in result["evidence"]:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        if ev["image_path"] and os.path.exists(ev["image_path"]):
                            st.image(ev["image_path"], use_column_width=True)
                        else:
                            st.write("Image not found")
                    with cols[1]:
                        st.write(f"**#{ev['rank']}** • Cosine: `{ev['cosine']:.3f}` • Prob: `{ev['prob']:.3f}`")
                        st.write(ev["caption"])
                        if ev.get("gt"):
                            st.caption(f"GT: {ev['gt']}")

            # debugging hints (collapsed)
            with st.expander("Debug / Tips"):
                st.write("If evidence seems unrelated, check:")
                st.write("- KB captions must accurately describe their images.")
                st.write("- Increase KB size with diverse, domain-specific examples.")
                st.write("- Tune α (blend) and threshold in the sidebar.")
                st.write("- You can inspect raw scores below.")
                st.json({
                    "pair_cos": result["pair_cos"],
                    "pair_prob": result["pair_prob"],
                    "kb_top_sim": result["kb_top_sim"],
                    "kb_top_prob": result["kb_top_prob"],
                    "confidence": result["confidence"]
                })

        except Exception as e:
            st.error(f"Failed to run prediction: {e}")
