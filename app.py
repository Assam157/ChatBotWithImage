from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HFACCESKEY")

# Real, open, working Hugging Face Spaces
HF_CHAT_URL = "https://abidlabs-chatgpt-mini.hf.space/run/predict"
HF_IMAGE_URL = "https://stabilityai-stable-diffusion-2.hf.space/run/predict"
HF_MODIFY_URL = "https://JonasKlose-StableDiffusionInpainting.hf.space/run/predict"

# =======================================================
# Chat Endpoint
# =======================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "Missing message"}), 400

        payload = {"data": [user_input]}
        response = requests.post(HF_CHAT_URL, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": f"HF Chat error: {response.text}"}), response.status_code

        result = response.json()
        reply = result.get("data", ["No response"])[0]
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# Image Generation Endpoint
# =======================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("message", "")
        if not prompt:
            return jsonify({"error": "Missing message"}), 400

        payload = {"data": [prompt]}
        response = requests.post(HF_IMAGE_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return jsonify({"error": f"HF Image error: {response.text}"}), response.status_code

        result = response.json()
        image_data = result.get("data", [None])[0]
        return jsonify({"image": image_data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# Image Modification (Inpainting)
# =======================================================
@app.route("/modify_image", methods=["POST"])
def modify_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url", "")
        instruction = data.get("instruction", "")
        if not image_url or not instruction:
            return jsonify({"error": "Missing fields"}), 400

        payload = {"data": [image_url, instruction]}
        response = requests.post(HF_MODIFY_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return jsonify({"error": f"HF Modify error: {response.text}"}), response.status_code

        result = response.json()
        modified_image = result.get("data", [None])[0]
        return jsonify({"modified_image": modified_image})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Backend running ✅"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
