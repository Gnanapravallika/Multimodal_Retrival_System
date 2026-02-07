import os
import faiss
import numpy as np
import torch
import constants
import utils
import embedding_utils
from tqdm import tqdm

def build_index_optimized():
    # 1. Load Model
    print("Loading model...")
    model, processor, tokenizer = embedding_utils.load_model()
    
    # 2. Load Dataset
    print("Loading captions and images...")
    try:
        df = utils.load_captions(constants.CAPTIONS_FILE)
    except FileNotFoundError:
        print("Dataset not found! Please ensure 'captions.txt' is in data/ and images are in data/Images/")
        return

    # Use a smaller batch size to prevent memory issues on CPU
    batch_size = 4 
    print(f"Creating DataLoader with batch_size={batch_size}...")
    transform = embedding_utils.get_transform(processor)
    dataloader = utils.create_dataloader(df, transform=transform, batch_size=batch_size)
    
    # 3. Generate Embeddings with Progress Bar
    print("Generating image embeddings... This may take a while on CPU.")
    
    model.eval()
    all_embeddings = []
    all_image_names = []
    
    # Custom loop with tqdm
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Images"):
            try:
                images = batch["image"].to(constants.DEVICE)
                image_names = batch["image_name"]
                
                # Get image features
                image_features = model.get_image_features(pixel_values=images)
                
                # Normalize embeddings
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                all_embeddings.append(image_features.cpu())
                all_image_names.extend(image_names)
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue
            
    if not all_embeddings:
        print("No embeddings generated. Exiting.")
        return

    image_embeddings = torch.cat(all_embeddings).numpy()
    
    # 4. Create FAISS Index
    print("Building FAISS index...")
    d = image_embeddings.shape[1] 
    index = faiss.IndexFlatIP(d) 
    index.add(image_embeddings)
    
    # 5. Save Artifacts
    if not os.path.exists(constants.EMBEDDINGS_DIR):
        os.makedirs(constants.EMBEDDINGS_DIR)
        
    print(f"Saving embeddings to {constants.EMBEDDINGS_DIR}...")
    np.save(constants.IMAGE_EMBEDDINGS_FILE, image_embeddings)
    np.save(os.path.join(constants.EMBEDDINGS_DIR, "image_names.npy"), all_image_names)
    faiss.write_index(index, constants.FAISS_INDEX_FILE)
    
    print("Index built and saved successfully!")

if __name__ == "__main__":
    build_index_optimized()
