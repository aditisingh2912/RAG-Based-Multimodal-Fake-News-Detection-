# RAG-Based-Multimodal-Fake-News-Detection
This repositiory is built for identifying Fake News from different modality like the images and text 
# Dataset
You can access the dataset here which is used for training [here](https://github.com/stevejpapad/image-text-verification/tree/master/VERITE)

# Architecture 
Hey all , I thought of putting a simple debugging architecture of Relevance Evidence Detection aka RED_DOT as I understand .Detailed explanation of the model along with base model definition can be found [here](https://github.com/stevejpapad/relevant-evidence-detection) along with the original paper .

The REDDOT model consists of a two-stage architecture:

## Transformer Encoder

1. Comprised of 4 MultiHead self-attention layers.

2. Accepts input tokens of shape [seq_len, embed_dim], where embed_dim = 768.

## Token Classifier

1. Operates on the output representation of the [CLS] token.

2. Predicts a binary label (real/fake) for the input evidence.


Here's a detailed explanation of how data flows through the REDDOT model during inference, with a focus on multimodal fusion, chunking, and transformer encoding for fake news detection.

We have two modalities intially - text and image.

a. After encoding we have image,text,fused embedding as [768,768,3840] in standard Pytorch standard of [batch_size,Seq_len,Embed_dimension]

b. Now our REDDOT encoder can accept embeddings with dimension [768] so this is where i realised we  need a chunking strategy for passing any input through the encoder architecture which might  look like [batch size, chhunks, embed_dim] ---> [5,768].

c. Lets see what is actually happening when we are creating chunks . Instead of directly passing , now we are passing 5 sequential tokens for each batch sample, with Shape as [1,5,768]. Now the self attention is computed between all the 5 tokens for the given tensor in the transformer architecture.

d. Also here each token_index comes with fusion embedding like concat_1, mul,add,sub and a prepended CLS token which is the contextual embedding. Output shape from transformer: [1, 6, 768]


# Environment 

Back-end Flask Server


To install the required Python libraries for the backend, run the command:
```bash 
 pip install -r requirements.txt
```
# Project Status
Right now I am working on preparing the inference pipeline for this project so come at the project after a while  .You might see something amazing , till then stay tuned .
To still give you a glimpse on it , here is the task list for inference pipeline

I have a pretrained open_source RED_DOT Model which you can acesss from [here](https://github.com/aditisingh2912/RAG-Based-Multimodal-Fake-News-Detection-/blob/main/Training%20Pipeline/models%20(1).py)REDDOT is an encoder only architecture .Traditionally have been trained on with encoder_version as ViT/L32 .My training pipeline uses the  Encoder_version = ViT/L14 . The base model was trained on VERITE dataset . 

The tasks are listed as follows:-
1. Load the pretrained RED_DOT model  using the provided checkpoints ,following parameters were used
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

