from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import os
from io import BytesIO

app = Flask(__name__)

# ✅ Allow CORS for your frontend
CORS(app, resources={r"/*": {"origins": "https://chatbotwithimagenew.onrender.com"}})

# ✅ DeepInfra API key (set in Render → Environment Variables)
DEEPINFRA_KEY = os.getenv("DEEPINFRA_KEY")

# ============================================================
# MODELS
# ============================================================
CHAT_MODEL = "meta-llama/Llama-3-8b-instruct"
IMG_GEN_MODEL = "stabilityai/stable-diffusion-2"
IMG_MODIFY_MODEL = "timbrooks/instruct-pix2pix"

BASE_URL_CHAT = "https://api.deepinfra.com/v1/openai/chat/completions"
BASE_URL_IMG = "https://api.deepinfra.com/v1/inference"

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
            "Authorization": f"Bearer {DEEPINFRA_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 200
        }

        response = requests.post(BASE_URL_CHAT, headers=headers, json=payload)

        if response.status_code != 200:
            return jsonify({"error": f"DeepInfra error: {response.text}"}), response.status_code

        reply = response.json()["choices"][0]["message"]["content"]
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

        headers = {"Authorization": f"Bearer {DEEPINFRA_KEY}"}
        payload = {"inputs": prompt}

        response = requests.post(
            f"{BASE_URL_IMG}/{IMG_GEN_MODEL}",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            return jsonify({"error": f"DeepInfra error: {response.text}"}), response.status_code

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

        headers = {"Authorization": f"Bearer {DEEPINFRA_KEY}"}
        files = {"image": (image.filename, image.stream, image.mimetype)}
        data = {"inputs": prompt}

        response = requests.post(
            f"{BASE_URL_IMG}/{IMG_MODIFY_MODEL}",
            headers=headers,
            data=data,
            files=files
        )

        if response.status_code != 200:
            return jsonify({"error": f"DeepInfra error: {response.text}"}), response.status_code

        image_bytes = response.content
        return send_file(BytesIO(image_bytes), mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "DeepInfra Flask Backend running successfully 🚀"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
