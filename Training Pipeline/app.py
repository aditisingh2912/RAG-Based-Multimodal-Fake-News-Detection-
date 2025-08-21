import os
import time
import faiss
import numpy as np
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# ---------------------------
# Path Handling (important!)
# ---------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))   # go one level up from Training_pipeline/
KB_DIR = os.path.join(ROOT_DIR, "knowledge_base")

# ---------------------------
# Load CLIP model
# ---------------------------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor

model, processor = load_model()

# ---------------------------
# Build FAISS index from KB
# ---------------------------
@st.cache_resource
def build_faiss_index():
    kb_embeddings = []
    kb_meta = []
    
    for folder in os.listdir(KB_DIR):
        folder_path = os.path.join(KB_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        
        img_path = os.path.join(folder_path, "Image.jpg")
        cap_path = os.path.join(folder_path, "caption.txt")
        gt_path = os.path.join(folder_path, "GT.txt")

        if not os.path.exists(img_path) or not os.path.exists(cap_path):
            continue

        # Load KB data
        image = Image.open(img_path).convert("RGB")
        with open(cap_path, "r") as f:
            caption = f.read().strip()
        gt_label = None
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                gt_label = f.read().strip()

        # Encode with CLIP
        inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model.get_text_features(**inputs).cpu().numpy()
        
        kb_embeddings.append(emb[0])
        kb_meta.append({
            "caption": caption,
            "image_path": img_path,
            "gt": gt_label
        })
    
    if len(kb_embeddings) == 0:
        return None, None

    # Build FAISS index
    dim = kb_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(kb_embeddings))

    return index, kb_meta

index, kb_meta = build_faiss_index()

# ---------------------------
# Prediction
# ---------------------------
def predict(text, image):
    start_time = time.time()

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_emb = model.get_text_features(**inputs).cpu().numpy()
        img_emb = model.get_image_features(**inputs).cpu().numpy()

    # Similarity between text & image
    sim = np.dot(text_emb, img_emb.T) / (np.linalg.norm(text_emb) * np.linalg.norm(img_emb))
    sim_score = float(sim[0][0])

    # Retrieve evidence from KB
    if index is not None:
        D, I = index.search(text_emb, k=1)
        evidence = kb_meta[I[0][0]]
    else:
        evidence = {"caption": "No KB available", "image_path": None, "gt": None}

    inference_time = time.time() - start_time
    label = "Real" if sim_score > 0.5 else "Fake"

    return label, sim_score, inference_time, evidence

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Fake Image-Text Detector", layout="centered")
st.title("🕵️ Fake or Real Detector (Image + Text)")

with st.form("user_input"):
    text_input = st.text_area("Enter description:")
    image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    if not text_input or not image_input:
        st.warning("Please provide both text and image.")
    else:
        image = Image.open(image_input).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, score, infer_time, evidence = predict(text_input, image)

        st.metric("Prediction", label, f"{score:.2f}")
        st.write(f"⏱ Inference Time: {infer_time:.2f} sec")
        st.write(f"🔎 Evidence: {evidence['caption']}")
        if evidence["image_path"] and os.path.exists(evidence["image_path"]):
            st.image(evidence["image_path"], caption="Retrieved Evidence", use_column_width=True)
        if evidence["gt"]:
            st.info(f"Ground Truth: {evidence['gt']}")

st.caption("⚡ Powered by Aditi Singh | Built for Streamlit Cloud")


