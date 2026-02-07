import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "flickr30k_images")
CAPTIONS_FILE = os.path.join(DATA_DIR, "results.csv") # Flickr30k usually has a results.csv

# Embeddings Paths
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
IMAGE_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "image_embeddings.npy")
TEXT_EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "text_embeddings.npy")
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, "faiss_index.bin")

# Model Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
DEVICE = "cpu"

# Search Configuration
TOP_K = 5
