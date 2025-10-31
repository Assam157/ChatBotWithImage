from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# ✅ Use the correct environment variable name (set this in Render)
HF_API_KEY = os.getenv("HFACCESKEY")

# ✅ Real Hugging Face model endpoints
HF_CHAT_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HF_MODIFY_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-inpainting"

# =======================================================
# 1️⃣ Chat Endpoint
# =======================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {"inputs": user_input}
        response = requests.post(HF_CHAT_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return jsonify({"error": f"HF Chat error: {response.text}"}), response.status_code

        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            reply = result[0]["generated_text"]
        else:
            reply = "No valid response received."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 2️⃣ Image Generation Endpoint
# =======================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("message", "")
        if not prompt:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {"inputs": prompt}
        response = requests.post(HF_IMAGE_URL, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"HF Image error: {response.text}"}), response.status_code

        result = response.json()
        # Some models return base64 image bytes in 'data' or 'generated_image'
        image_data = result if isinstance(result, list) else result.get("data", [None])[0]
        return jsonify({"image": image_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 3️⃣ Image Modification (Inpainting)
# =======================================================
@app.route("/modify_image", methods=["POST"])
def modify_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url", "")
        instruction = data.get("instruction", "")
        if not image_url or not instruction:
            return jsonify({"error": "Missing fields"}), 400

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {"inputs": {"image": image_url, "prompt": instruction}}
        response = requests.post(HF_MODIFY_URL, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"HF Modify error: {response.text}"}), response.status_code

        result = response.json()
        modified_image = result if isinstance(result, list) else result.get("data", [None])[0]
        return jsonify({"modified_image": modified_image})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 4️⃣ Health Check
# =======================================================
@app.route("/")
def home():
    return jsonify({"status": "Backend running ✅"})


# =======================================================
# Run Server
# =======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
