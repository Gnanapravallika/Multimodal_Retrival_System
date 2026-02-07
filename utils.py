import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import constants

def load_captions(captions_file):
    """
    Loads captions from the Flickr8k captions.txt file.
    Format: image,caption
    """
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found at {captions_file}")
    
    df = pd.read_csv(captions_file)
    # Ensure columns are correct (image, caption)
    # Flickr8k usually has header: image,caption
    return df

def get_image_path(image_name):
    """Returns the full path for an image."""
    return os.path.join(constants.IMAGES_DIR, image_name)

class FlickrDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.image_names = df['image'].unique() # Unique images
        
        # Create a mapping of image_name -> list of captions
        self.captions_map = df.groupby('image')['caption'].apply(list).to_dict()
        
        self.all_images = list(self.image_names)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = get_image_path(image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy black image in case of error
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
        
        captions = self.captions_map[image_name]
        
        return {
            "image": image,
            "image_name": image_name,
            "captions": captions
        }

def create_dataloader(df, transform, batch_size=32):
    dataset = FlickrDataset(df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
