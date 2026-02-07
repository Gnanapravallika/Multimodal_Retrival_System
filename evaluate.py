import numpy as np
import torch
import constants
import utils
import embedding_utils
import search
from tqdm import tqdm
import os

def evaluate(k_values=[1, 5, 10]):
    print("Initialize Evaluation...")
    # Load model and data
    model, processor, tokenizer = embedding_utils.load_model()
    
    # We will use a subset for quick evaluation or full set if possible
    try:
        captions_df = utils.load_captions(constants.CAPTIONS_FILE)
    except FileNotFoundError:
        print("Captions file not found.")
        return

    # Sample queries for evaluation
    sample_queries = captions_df['caption'].sample(100).tolist()
    print(f"Evaluating on {len(sample_queries)} random queries...")
    
    k_values = [1, 5, 10]
    recalls = {k: 0 for k in k_values}
    
    # Latency Tracking
    latencies = []
    
    model.eval()
    
    start_time = time.time()
    
    for query in sample_queries:
        t0 = time.time()
        
        # 1. Generate text embedding
        text_emb = embedding_utils.generate_text_embedding(model, tokenizer, [query])
        
        # 2. Search
        distances, indices = index.search(text_emb, max(k_values))
        
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
        
        # Recall calculation (simplified for this demo context)
        # Ideally we check if the retrieved image matches the caption's image ID
        # For this system to be "real", we need image_id mapping. 
        # Assuming we just measure search speed for now if ground truth isn't perfect.
        
    total_time = time.time() - start_time
    avg_latency = np.mean(latencies)
    throughput = len(sample_queries) / total_time
    
    print("\nXXX SYSTEM METRICS XXX")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} QPS")
    print(f"Peak Memory Usage: {mem_after:.2f} MB")
    print(f"Index Size: {index.ntotal} vectors")
    
    # Note: Accuracy metrics (Recall@K) require strict ground truth mapping 
    # which depends on the exact dataset structure (Flickr30k).
    # Since we updated to Flickr30k, we'd need to parse its specific format to match 
    # specific image filenames to captions for Recall. 
    # For now, we focus on the requested SYSTEM metrics.

if __name__ == "__main__":
    evaluate_system()
