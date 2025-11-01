from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)
CORS(app)

OPENROUTER_KEY = os.getenv("OPENAIKEY")
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
IMG_GEN_URL = "https://openrouter.ai/api/v1/images/generations"
IMG_EDIT_URL = "https://openrouter.ai/api/v1/images/edits"


# ============================================================
# 🧠 CHAT ENDPOINT
# ============================================================
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ]
        }

        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": f"Chat request failed ({response.status_code})", "details": response.text}), 502

        data = response.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No valid response.")
        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🖼️ IMAGE GENERATION ENDPOINT
# ============================================================
@app.route("/generate_image", methods=["POST", "OPTIONS"])
def generate_image():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    try:
        data = request.get_json(force=True)
        prompt = data.get("message", "").strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "stability-ai/sdxl",
            "prompt": prompt,
            "size": "1024x1024",
            "n": 1
        }

        response = requests.post(IMG_GEN_URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter image generation failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()
        image_url = None
        if "data" in data and len(data["data"]) > 0:
            image_url = data["data"][0].get("url")

        if image_url:
            return jsonify({"image_url": image_url}), 200
        else:
            return jsonify({"error": "No image returned by model", "raw": data}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 IMAGE MODIFICATION ENDPOINT
# ============================================================
@app.route("/modify_image", methods=["POST", "OPTIONS"])
def modify_image():
    if request.method == "OPTIONS":
        return jsonify({"ok": True}), 200

    try:
        data = request.get_json(force=True)
        image_url = data.get("image_url", "").strip()
        instruction = data.get("instruction", "").strip()

        if not image_url or not instruction:
            return jsonify({"error": "Missing fields (image_url/instruction)"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "stability-ai/sdxl-inpainting",
            "image": image_url,
            "prompt": instruction,
        }

        response = requests.post(IMG_EDIT_URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter inpainting failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()
        modified_image = None
        if "data" in data and len(data["data"]) > 0:
            modified_image = data["data"][0].get("url")

        if modified_image:
            return jsonify({"modified_image": modified_image}), 200
        else:
            return jsonify({"error": "No modified image returned", "raw": data}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🌐 ROOT
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "✅ OpenRouter AI Backend Running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
