from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import search
from PIL import Image
import logging

from logger_config import setup_logger

app = Flask(__name__)

# Configure centralized logging
logger = setup_logger(__name__)


# Initialize Search Engine
engine = search.SearchEngine()
try:
    engine.load_resources()
except Exception as e:
    logger.error(f"Error loading resources: {e}")
    logger.warning("Ensure you have run build_index.py first!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring."""
    status = "healthy" if engine.is_loaded else "not_loaded"
    return jsonify({"status": status, "service": "multimodal-search"}), 200

@app.route('/api/search', methods=['POST'])
def search_api():
    """
    Unified search endpoint.
    Expects JSON for text search: {"query": "text...", "k": 5}
    Expects Multipart/Form-Data for image search: file=..., k=...
    """
    if not engine.is_loaded:
        return jsonify({"error": "Search index not loaded"}), 503

    k = 5
    results = []

    try:
        # Check if it's an image upload
        if 'file' in request.files:
            file = request.files['file']
            k = int(request.form.get('k', 5))
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            img = Image.open(file).convert("RGB")
            results = engine.search_image(img, k=k)
            
        # Check if it's a text query (JSON)
        elif request.is_json:
            data = request.json
            query = data.get('query', '')
            k = int(data.get('k', 5))
            
            if not query:
                return jsonify({"error": "No query provided"}), 400
            
            results = engine.search_text(query, k=k)
            
        else:
            return jsonify({"error": "Unsupported media type or missing data"}), 415

        # Format results
        json_results = []
        for path, score in results:
            filename = os.path.basename(path)
            json_results.append({
                "filename": filename,
                "score": float(score),
                "url": f"/images/{filename}"
            })
            
        return jsonify({"results": json_results})

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

# Route to serve images from the data directory
@app.route('/images/<filename>')
def serve_image(filename):
    import constants
    return send_from_directory(constants.IMAGES_DIR, filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
