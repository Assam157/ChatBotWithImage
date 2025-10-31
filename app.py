from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # ✅ Allow frontend access

HF_API_KEY = os.getenv("HFACCESKEY")  # Optional Hugging Face API Key


# =======================================================
# 1️⃣ Chat Endpoint — Using DialoGPT (Free HF Space)
# =======================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")

        if not user_input:
            return jsonify({"error": "Missing message"}), 400

        # ✅ Public HF Space that allows chat
        hf_url = "https://huggingface.co/spaces/abidlabs/ChatGPT-mini/api/predict/"

        payload = {"data": [user_input]}
        headers = {"Content-Type": "application/json"}

        response = requests.post(hf_url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace chat error: {response.text}"}), response.status_code

        result = response.json()
        reply = None

        if isinstance(result, dict) and "data" in result:
            reply = result["data"][0]
        else:
            reply = "No valid response received."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 2️⃣ Image Generation Endpoint — Stable Diffusion (Free)
# =======================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "Missing message"}), 400

        payload = {"data": [message]}
        headers = {"Content-Type": "application/json"}

        # ✅ Free public Stable Diffusion HF space
        hf_url = "https://huggingface.co/spaces/stabilityai/stable-diffusion-2/api/predict/"

        response = requests.post(hf_url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace image gen error: {response.text}"}), response.status_code

        result = response.json()
        image_url = None

        if isinstance(result, dict) and "data" in result:
            image_url = result["data"][0]
        else:
            image_url = "No image returned."

        return jsonify({"image_url": image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 3️⃣ Image Modification (Inpainting / Edit)
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
        headers = {"Content-Type": "application/json"}

        hf_url = "https://huggingface.co/spaces/JonasKlose/StableDiffusionInpainting/api/predict/"

        response = requests.post(hf_url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace modify error: {response.text}"}), response.status_code

        result = response.json()
        modified_url = None

        if isinstance(result, dict) and "data" in result:
            modified_url = result["data"][0]
        else:
            modified_url = "No modified image returned."

        return jsonify({"modified_image_url": modified_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# Root route
# =======================================================
@app.route("/")
def home():
    return jsonify({"status": "Backend is running ✅"})


# =======================================================
# Run Server
# =======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
