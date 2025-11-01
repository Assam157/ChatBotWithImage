from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# ✅ Environment variables
OPENROUTER_KEY = os.getenv("OPENAiKey")
FALAI_KEY = os.getenv("FalAIKey")

# ✅ API URLs
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FALAI_GEN_URL = "https://fal.run/fal-ai/flux-pro"
FALAI_MOD_URL = "https://fal.run/fal-ai/flux-pro-inpainting"

# =======================================================
# 1️⃣ CHAT — OpenRouter
# =======================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "")
        if not message:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": message}
            ],
            "temperature": 0.7,
            "max_tokens": 200
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            reply = result["choices"][0]["message"]["content"]
        else:
            reply = result.get("error", "No response received")

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 2️⃣ IMAGE GENERATION — Fal.ai
# =======================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("message", "")
        if not prompt:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Key {FALAI_KEY}",
            "Content-Type": "application/json"
        }

        payload = {"prompt": prompt}
        response = requests.post(FALAI_GEN_URL, headers=headers, json=payload, timeout=120)
        result = response.json()

        if "images" in result and len(result["images"]) > 0:
            image_url = result["images"][0]["url"]
            return jsonify({"image_url": image_url})
        else:
            return jsonify({"error": result}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 3️⃣ IMAGE MODIFICATION — Fal.ai Inpainting
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
            "Authorization": f"Key {FALAI_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "image_url": image_url,
            "prompt": instruction
        }

        response = requests.post(FALAI_MOD_URL, headers=headers, json=payload, timeout=120)
        result = response.json()

        if "images" in result and len(result["images"]) > 0:
            modified_image_url = result["images"][0]["url"]
            return jsonify({"modified_image_url": modified_image_url})
        else:
            return jsonify({"error": result}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================================================
# 4️⃣ HEALTH CHECK
# =======================================================
@app.route("/")
def home():
    return jsonify({"status": "Backend running with Fal.ai + OpenRouter ✅"})


# =======================================================
# Run Server
# =======================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
