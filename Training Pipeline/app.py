import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="Fake or Real Detector", layout="centered")

st.title("🕵️ Fake or Real Detector (Text + Image)")

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

model, processor = load_model()

def predict_similarity(text, image):
    # Add a "negative" candidate
    candidate_texts = [text, "This description does not match the image"]

    inputs = processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)


    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1)     

    similarity_score = probs[0][0].item()  
    return similarity_score

with st.form("input_form"):
    text_input = st.text_area("Enter the description for the image", "")
    image_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    submit_btn = st.form_submit_button("Predict")

if submit_btn:
    if not text_input or not image_input:
        st.warning("Please provide both text and image.")
    else:
        image = Image.open(image_input).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        similarity = predict_similarity(text_input, image)
        label = "Real" if similarity > 0.5 else "Fake"
        st.metric("Prediction", label)
        st.write(f"**Similarity Score:** {similarity:.4f}")
        st.info("A higher similarity score means the text is likely describing the image correctly.")

st.markdown("---")
st.caption("Powered by OpenAI CLIP & Hugging Face | Created by [AditiSingh004](https://github.com/AditiSingh004)")
