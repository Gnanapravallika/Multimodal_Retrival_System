# 🚀 Running Multimodal Retrieval System on Google Colab

This guide explains how to run this project on Google Colab using a GPU.

## 1. Setup in Colab

Open [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 1: Clone the Repository
Run this cell to clone your code:
```python
!git clone https://github.com/Gnanapravallika/Multimodal_Retrival_System.git
%cd Multimodal_Retrival_System
```

### Step 2: Install Dependencies
```python
!pip install -r requirements.txt
!pip install pyngrok  # For exposing the web app
```

### Step 3: Get the Dataset
Since the dataset is not on GitHub, you have two options:

#### Option A: Download from Kaggle (Recommended)
1. Upload your `kaggle.json` key to Colab files.
2. Run:
```python
import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content"
# This downloads the Flickr30k dataset
!kaggle datasets download -d hsankesara/flickr-image-dataset
!unzip -q flickr-image-dataset.zip -d data/
# Rename folder to match code expectation if needed
import os
if os.path.exists("data/flickr30k_images"):
   print("Dataset ready.")
else:
   # Adjust this move command based on actual unzip structure
   !mv data/flickr30k_images/flickr30k_images/* data/flickr30k_images/
```

#### Option B: Google Drive
1. Upload your `flickr30k_images` folder to Google Drive.
2. Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
# Symlink to project data folder
!ln -s "/content/drive/MyDrive/path_to_your_dataset/flickr30k_images" data/flickr30k_images
```

### Step 4: Build Index
Generate embeddings (this needs GPU!).
```python
!python build_index.py
```

### Step 5: Run the App
We use `pyngrok` to create a public URL for the Flask app.

1.  **Sign up** at [ngrok.com](https://dashboard.ngrok.com/signup) and get your Authtoken.
2.  Run this cell:

```python
from pyngrok import ngrok

# Authenticate
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

# Run Flask in background
get_ipython().system_raw("python main.py &")

# Open Tunnel
public_url = ngrok.connect(5000).public_url
print(f"🚀 App works at: {public_url}")
```
Click the link to start searching!
