from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # ✅ Allow frontend access


# =======================================================
# 1️⃣ Chat Endpoint — Using DialoGPT (No Token Needed)
# =======================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("prompt", "")

        payload = {"data": [user_input]}
        response = requests.post(
            "https://huggingface.co/spaces/microsoft/DialoGPT-medium/api/predict/",
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace chat error: {response.text}"}), 500

        result = response.json()
        reply = result.get("data", ["No reply"])[0]
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 2️⃣ Image Generation Endpoint — Stable Diffusion (No Token)
# =======================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        payload = {"data": [prompt]}
        response = requests.post(
            "https://huggingface.co/spaces/stabilityai/stable-diffusion/api/predict/",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace image gen error: {response.text}"}), 500

        result = response.json()
        image_url = result.get("data", [None])[0]
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

        payload = {"data": [image_url, instruction]}
        response = requests.post(
            "https://huggingface.co/spaces/JonasKlose/StableDiffusionInpainting/api/predict/",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace modify error: {response.text}"}), 500

        result = response.json()
        modified_url = result.get("data", [None])[0]
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

