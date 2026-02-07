print("Importing os...")
import os
print("Importing faiss...")
import faiss
print("Importing numpy...")
import numpy as np
print("Importing torch...")
import torch
print("Importing constants...")
import constants
print("Importing utils...")
import utils
print("Importing embedding_utils...")
import embedding_utils
print("Imports done.")

def build_index():
    # 1. Load Model
    model, processor, tokenizer = embedding_utils.load_model()
    
    # 2. Load Dataset
    print("Loading captions and images...")
    try:
        df = utils.load_captions(constants.CAPTIONS_FILE)
    except FileNotFoundError:
        print("Dataset not found! Please ensure 'captions.txt' is in data/ and images are in data/Images/")
        return

    transform = embedding_utils.get_transform(processor)
    dataloader = utils.create_dataloader(df, transform=transform, batch_size=32)
    
    # 3. Generate Embeddings
    print("Generating image embeddings... This may take a while on CPU.")
    image_embeddings, image_names = embedding_utils.generate_image_embeddings(model, dataloader)
    
    # 4. Create FAISS Index
    print("Building FAISS index...")
    d = image_embeddings.shape[1] # Dimension (512 for CLIP ViT-B/32)
    
    # IndexFlatIP = Inner Product (exact search). Normalize vectors before adding.
    index = faiss.IndexFlatIP(d) 
    index.add(image_embeddings)
    
    # 5. Save Artifacts
    if not os.path.exists(constants.EMBEDDINGS_DIR):
        os.makedirs(constants.EMBEDDINGS_DIR)
        
    print(f"Saving embeddings to {constants.EMBEDDINGS_DIR}...")
    np.save(constants.IMAGE_EMBEDDINGS_FILE, image_embeddings)
    # Save image names (metadata) to allow mapping index -> filename
    # We'll save it as a simple text file or numpy array
    np.save(os.path.join(constants.EMBEDDINGS_DIR, "image_names.npy"), image_names)
    
    faiss.write_index(index, constants.FAISS_INDEX_FILE)
    
    print("Index built and saved successfully!")

if __name__ == "__main__":
    build_index()
