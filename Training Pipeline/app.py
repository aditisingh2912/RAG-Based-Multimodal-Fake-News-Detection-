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
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-large-patch14"

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

        # Load caption
        with open(cap_path, "r") as f:
            caption = f.read().strip()
        gt_label = None
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                gt_label = f.read().strip()

        # ✅ Only encode text (not images)
        text_inputs = processor(text=[caption], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model.get_text_features(**text_inputs).cpu().numpy()
        
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

    # Encode input text
    text_inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    text_emb = model.get_text_features(**text_inputs).cpu().detach().numpy()

    # Encode input image
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    image_emb = model.get_image_features(**image_inputs).cpu().detach().numpy()

    # Similarity between text & image (cosine similarity)
    sim = cos_sim(torch.tensor(text_emb), torch.tensor(image_emb)).item()

    # Retrieve top-3 evidence from KB
    D, I = index.search(text_emb.astype("float32"), k=3)
    evidence = [kb_meta[i] for i in I[0]]

    # Label decision
    label = "Real" if sim > 0.5 else "Fake"

    end_time = time.time()
    infer_time = round(end_time - start_time, 3)

    return label, round(sim, 3), infer_time, evidence



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


