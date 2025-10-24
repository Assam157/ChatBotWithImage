import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your API keys
HF_API_KEY = os.getenv("HFACCESKEY")  # Hugging Face
API_KEY = "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu"  # DeepInfra/OpenAI

# Hugging Face endpoint for Stable Diffusion
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"

# DeepInfra/OpenAI endpoint for chat
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ======== IMAGE GENERATION ========
@app.route("/Image", methods=["POST"])
def Image():
    data = request.json
    prompt = data.get("message", "").strip()

    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    response = requests.post(HF_IMAGE_URL, json=payload, headers=headers)

    if response.status_code == 200:
        # Hugging Face can return JSON with base64-encoded image or raw bytes
        try:
            # Try decoding JSON base64 response
            image_base64 = response.json()[0]["generated_image"]
            image_data = base64.b64decode(image_base64)
        except Exception:
            # Fallback: raw bytes
            image_data = response.content

        # Save image with unique filename
        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join("static", filename)

        with open(image_path, "wb") as f:
            f.write(image_data)

        # Return URL for accessing the image
        return jsonify({"image_url": f"https://chatbotwithimagebackend.onrender.com/static/{filename}"})
    else:
        return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

# ======== CHAT ========
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip()

    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # Change if needed
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(CHAT_URL, json=payload, headers=headers)

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply})
    else:
        return jsonify({"error": response.json()}), response.status_code

# ======== RUN APP ========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)

