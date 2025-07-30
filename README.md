# RAG-Based-Multimodal-Fake-News-Detection
This repositiory is built for identifying Fake News from different modality like the images and text 
# Dataset
You can access the dataset here which is used for training [here](https://github.com/stevejpapad/image-text-verification/tree/master/VERITE)
# Environment Setup

Back-end Flask Server


To install the required Python libraries for the backend, run the command:
```bash 
 pip install -r requirements.txt
```
# Project Status
Right now I am working on preparing the inference pipeline for this project so come at the project after a while  .You might see something amazing , till then stay tuned .
To still give you a glimpse on it , here is the task list for inference pipeline

I have a pretrained open_source RED_DOT Model which you can acesss from [here](https://github.com/aditisingh2912/RAG-Based-Multimodal-Fake-News-Detection-/blob/main/Training%20Pipeline/models%20(1).py)REDDOT is an encoder only architecture .Traditionally have been trained on .My training pipeline uses the  Encoder_version = ViT/L14 . The base model was trained on VERITE dataset . 

The tasks are listed as follows:-
1. Load the pretrained RED_DOT model ,following parameters were used
   ```bash
   a.  emb_dim=768, #originally 512
   b.  tf_layers=4, 
   c.  tf_head=8,
   d.  tf_dim=128,
   e.  use_evidence=0,
   f.  use_neg_evidence=0,
   ```
2. Preprocess and perform feature extraction for a single text and image using OPEN AI CLIP  and load_and_preprocess model from LAVIS library.Please install the defined versions from the Requirement.txt
   
3. Extract embeddings using CLIP model= ViT/L14
4. Build a FAISS vectorDB to map the embeddings with the query vector for obtaining search similarity based on neighbour indices
5. Verify the indices using a VERITE_Evidence.csv file
6. Feed the forward Method of RED_DOT model with evidence embeddings and the user query image embedding and user query text embedding
7. Calculate Confidence ,entropy score and classify the user input(text+image) as Real or Fake

