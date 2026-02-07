import faiss
import numpy as np
import os
import constants
import embedding_utils
from PIL import Image
from logger_config import setup_logger

logger = setup_logger(__name__)


class SearchEngine:
    def __init__(self):
        self.index = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_names = None
        self.is_loaded = False

    def load_resources(self):
        """Loads the FAISS index, metadata, and CLIP model."""
        if self.is_loaded:
            return


        logger.info("Loading search resources...")
        
        # Load Index
        if os.path.exists(constants.FAISS_INDEX_FILE):
            self.index = faiss.read_index(constants.FAISS_INDEX_FILE)
        else:
            logger.error("FAISS index not found.")
            raise FileNotFoundError("FAISS index not found. Run build_index.py first.")
            
        # Load Metadata
        names_path = os.path.join(constants.EMBEDDINGS_DIR, "image_names.npy")
        if os.path.exists(names_path):
            self.image_names = np.load(names_path)
        else:
            logger.error("Image names metadata not found.")
            raise FileNotFoundError("Image names metadata not found.")

        # Load Model (cached)
        self.model, self.processor, self.tokenizer = embedding_utils.load_model()
        self.is_loaded = True
        logger.info("Resources loaded successfully.")


    def search_text(self, text_query, k=5):
        """
        Searches for images matching the text query.
        Returns: proper image paths and distances
        """
        if not self.is_loaded:
            self.load_resources()
            
        # Generate text embedding
        text_emb = embedding_utils.generate_text_embedding(
            self.model, self.tokenizer, [text_query]
        )
        
        # Search
        distances, indices = self.index.search(text_emb, k)
        
        return self._format_results(distances[0], indices[0])

    def search_image(self, image_input, k=5):
        """
        Searches for images similar to the input image.
        image_input: PIL Image or path
        """
        if not self.is_loaded:
            self.load_resources()

        # Load image if path
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input

        # Preprocess using the raw processor (not the dataloader transform)
        inputs = self.processor(images=image, return_tensors="pt").to(constants.DEVICE)
        
        # specific embedding generation for single image
        import torch
        with torch.no_grad():
            img_emb = self.model.get_image_features(**inputs)
            img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
            img_emb = img_emb.cpu().numpy()

        # Search
        distances, indices = self.index.search(img_emb, k)
        
        return self._format_results(distances[0], indices[0])

    def _format_results(self, distances, indices):
        """Helper to map indices to filenames."""
        results = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.image_names):
                fname = self.image_names[idx]
                fpath = os.path.join(constants.IMAGES_DIR, fname)
                results.append((fpath, float(dist)))
        return results
