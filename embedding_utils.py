import torch
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import constants

def load_model():
    """Loads the CLIP model, processor, and tokenizer."""
    print(f"Loading model: {constants.MODEL_NAME}...")
    model = CLIPModel.from_pretrained(constants.MODEL_NAME).to(constants.DEVICE)
    processor = CLIPProcessor.from_pretrained(constants.MODEL_NAME)
    tokenizer = CLIPTokenizer.from_pretrained(constants.MODEL_NAME)
    print("Model loaded successfully.")
    return model, processor, tokenizer

def get_transform(processor):
    """
    Returns a transform function acting as a wrapper for the processor's image processing.
    This allows using the HuggingFace processor within a PyTorch Dataset/DataLoader.
    """
    # The processor returns a dict with 'pixel_values', we need to extract that tensor.
    # We'll create a callable that takes a PIL image and returns the tensor.
    
    def transform(image):
        # inputs is a dict: {'pixel_values': tensor}
        inputs = processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0) # Remove batch dimension added by processor
    
    return transform

def generate_image_embeddings(model, dataloader):
    """
    Generates embeddings for all images in the dataloader.
    Returns:
        image_embeddings: numpy array
        image_names: list of filenames
    """
    model.eval()
    all_embeddings = []
    all_image_names = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(constants.DEVICE)
            image_names = batch["image_name"]
            
            # Get image features
            image_features = model.get_image_features(pixel_values=images)
            
            # Normalize embeddings (important for cosine similarity / dot product)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            all_embeddings.append(image_features.cpu())
            all_image_names.extend(image_names)
            
    return torch.cat(all_embeddings).numpy(), all_image_names

def generate_text_embedding(model, tokenizer, text):
    """
    Generates embedding for a single query text (or list of texts).
    """
    model.eval()
    inputs = tokenizer(text, padding=True, return_tensors="pt").to(constants.DEVICE)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # Normalize
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
    return text_features.cpu().numpy()
