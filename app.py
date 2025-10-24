import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Hugging Face API Key
HF_API_KEY = os.getenv("HFACCESKEY")
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"

@app.route("/image", methods=["POST"])
def image():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}

    try:
        # Call Hugging Face API
        response = requests.post(HF_IMAGE_URL, json=payload, headers=headers, timeout=60)

        if response.status_code != 200:
            return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

        # Hugging Face can return either:
        # 1. JSON with base64 under "generated_image"
        # 2. Raw bytes (image)
        try:
            image_base64 = response.json()[0]["generated_image"]
        except Exception:
            image_base64 = base64.b64encode(response.content).decode()

        return jsonify({"image_base64": image_base64}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======== RUN APP ========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

