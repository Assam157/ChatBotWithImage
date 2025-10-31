from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# =======================================================
# 🔐 Secure API Config
# =======================================================
HF_API_KEY = os.getenv("HF_API_KEY") or "your_huggingface_token_here"
HEADERS = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# Model endpoints
CHAT_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
IMAGE_GEN_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
INPAINT_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

# =======================================================
# 💬 CHAT ENDPOINT
# =======================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "Missing message"}), 400

        payload = {"inputs": message}
        response = requests.post(CHAT_URL, headers=HEADERS, json=payload, timeout=60)

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace chat error: {response.text}"}), response.status_code

        result = response.json()
        # Extract model reply
        reply = result[0]["generated_text"] if isinstance(result, list) and len(result) > 0 else "No reply"
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 🖼️ IMAGE GENERATION ENDPOINT
# =======================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("message", "")
        if not prompt:
            return jsonify({"error": "Missing message"}), 400

        payload = {"inputs": prompt}
        response = requests.post(IMAGE_GEN_URL, headers=HEADERS, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace image gen error: {response.text}"}), response.status_code

        image_data = response.content
        # Return image bytes in base64 if needed
        import base64
        image_b64 = base64.b64encode(image_data).decode("utf-8")

        return jsonify({"image_base64": image_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 🧠 IMAGE MODIFICATION (INPAINTING)
# =======================================================
@app.route("/modify_image", methods=["POST"])
def modify_image():
    try:
        data = request.get_json()
        image_b64 = data.get("image_base64", "")
        prompt = data.get("instruction", "")

        if not image_b64 or not prompt:
            return jsonify({"error": "Missing fields"}), 400

        import base64
        image_bytes = base64.b64decode(image_b64)
        files = {"file": ("image.png", image_bytes, "image/png")}
        payload = {"inputs": prompt}

        response = requests.post(INPAINT_URL, headers={"Authorization": f"Bearer {HF_API_KEY}"}, files=files, data=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": f"HuggingFace modify error: {response.text}"}), response.status_code

        mod_image_data = response.content
        mod_image_b64 = base64.b64encode(mod_image_data).decode("utf-8")
        return jsonify({"modified_image_base64": mod_image_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 🏠 ROOT ROUTE
# =======================================================
@app.route("/")
def home():
    return jsonify({"status": "✅ Flask backend is running securely!"})


# =======================================================
# 🚀 RUN SERVER
# =======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
