from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import os
from io import BytesIO

app = Flask(__name__)

# ✅ Allow CORS for your frontend domain
CORS(app, resources={r"/*": {"origins": "https://chatbotwithimagenew.onrender.com"}})

# ✅ Hugging Face Access Token (set in Render Environment Variables)
HF_TOKEN = os.getenv("HFACCESSTOKEN")

# ✅ Model Endpoints
CHAT_MODEL = "HuggingFaceH4/zephyr-7b-beta"
IMG_GEN_MODEL = "stabilityai/stable-diffusion-2"
IMG_MODIFY_MODEL = "timbrooks/instruct-pix2pix"

# ============================================================
# CHAT ENDPOINT
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "No message provided"}), 400

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Accept": "application/json"
        }

        payload = {
            "inputs": message,
            "parameters": {"max_new_tokens": 200},
        }

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{CHAT_MODEL}",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return jsonify({"error": f"HF error: {response.text}"}), response.status_code

        data = response.json()
        # Some models return a list, handle both
        if isinstance(data, list) and len(data) > 0:
            reply = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            reply = data.get("generated_text", "")
        else:
            reply = "Sorry, I couldn't generate a response."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# IMAGE GENERATION ENDPOINT
# ============================================================
@app.route("/image_generate", methods=["POST"])
def image_generate():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": prompt}

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{IMG_GEN_MODEL}",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return jsonify({"error": f"HF error: {response.text}"}), response.status_code

        image_bytes = response.content
        return send_file(BytesIO(image_bytes), mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# IMAGE MODIFICATION ENDPOINT
# ============================================================
@app.route("/image_modify", methods=["POST"])
def image_modify():
    try:
        prompt = request.form.get("prompt")
        image = request.files.get("image")

        if not image or not prompt:
            return jsonify({"error": "Prompt and image file required"}), 400

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        files = {
            "image": (image.filename, image.stream, image.mimetype),
        }
        data = {"inputs": prompt}

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{IMG_MODIFY_MODEL}",
            headers=headers,
            data=data,
            files=files,
        )

        if response.status_code != 200:
            return jsonify({"error": f"HF error: {response.text}"}), response.status_code

        image_bytes = response.content
        return send_file(BytesIO(image_bytes), mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask HF Backend running successfully 🚀"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
