# AI-Powered Multimodal Retrieval System - User Guide

## 1. Prerequisites
Ensure you have installed the requirements:
```bash
pip install -r requirements.txt
```

## 2. Dataset Setup
1.  Download the **Flickr8k** dataset (from Kaggle or other sources).
2.  Extract the images into: `d:\Multimodal Retrieval System\data\Images\`
3.  Place `captions.txt` into: `d:\Multimodal Retrieval System\data\`

Structure should look like:
```
data/
├── Images/
│   ├── 1000268201_693b08cb0e.jpg
│   ├── ...
└── captions.txt
```

## 3. Build the Index (One-Time Setup)
Run this script to generate embeddings and create the FAISS index. This ensures searches are fast.
```bash
python build_index.py
```
*Note: This might take 10-20 minutes depending on your CPU.*

## 4. Run the Application
### Option A: Streamlit UI
Launch the easy-to-use Streamlit interface:
```bash
streamlit run app.py
```

### Option B: HTML/CSS Interface (Flask)
Run the custom web interface:
```bash
python flask_app.py
```
Then open your browser at: `http://127.0.0.1:5000`

## 5. Evaluate (Optional)
Check the system's performance:
```bash
python evaluate.py
```

## 6. Run on Google Colab
If you want to run this project on Google Colab (free GPU), read the [Colab Guide](COLAB_GUIDE.md).

