# How to Run the Multimodal Retrieval System

Follow these steps to set up and run the application.

## 1. Prerequisites
Ensure you have Python installed. Install the required libraries:
```bash
pip install -r requirements.txt
```

## 2. Dataset Setup (Flickr8k)
You need to download the Flickr8k dataset manually.
1.  **Download Images**: Get the Flickr8k images and extract them to:
    `d:\Multimodal Retrieval System\data\Images\`
    (The folder should contain .jpg files directly).

2.  **Download Captions**: Get `captions.txt` and place it in:
    `d:\Multimodal Retrieval System\data\`

**Structure Check:**
```
d:\Multimodal Retrieval System\
├── data\
│   ├── Images\
│   │   ├── 1000268201_693b08cb0e.jpg
│   │   └── ...
│   └── captions.txt
```

## 3. Build the Index (One-Time Setup)
Run the optimized script to generate embeddings. This version includes a progress bar and uses less memory.
```bash
python build_index_optimized.py
```
*Note: This usually takes 10-20 minutes depending on your CPU.*

## 4. Run the Application
### Option A: Web Interface (Flask)
Run the custom web interface:
```bash
python flask_app.py
```
Then open your browser at: `http://127.0.0.1:5000`

### Option B: Streamlit UI (Simpler)
Launch the easy-to-use Streamlit interface:
```bash
streamlit run app.py
```

## 5. Verify Installation
Run the evaluation script to check performance:
```bash
python evaluate.py
```
