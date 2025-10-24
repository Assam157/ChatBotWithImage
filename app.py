import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ======== API KEYS ========
HF_API_KEY = os.getenv("HFACCESKEY")  # Hugging Face
API_KEY = "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu"  # DeepInfra/OpenAI

HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ======== IMAGE GENERATION (BASE64) ========
@app.route("/image", methods=["POST"])
def image():
    data = request.json
    prompt = data.get("message", "").strip()
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}

    response = requests.post(HF_IMAGE_URL, json=payload, headers=headers)
    if response.status_code == 200:
        try:
            # Hugging Face may return JSON with base64
            image_base64 = response.json()[0]["generated_image"]
        except Exception:
            # fallback: encode raw bytes as base64
            image_base64 = base64.b64encode(response.content).decode()

        # Return base64 string directly
        return jsonify({"image_base64": image_base64})
    else:
        return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

# ======== CHAT ========
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip()
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
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
    app.run(debug=True, host="0.0.0.0", port=5000)

