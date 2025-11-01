from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# === Environment Key ===
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

# === API URL ===
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ============================================================
# 🧠 CHAT — OpenRouter
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_input = data.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": f"OpenRouter request failed ({response.status_code})", "details": response.text}), 502

        data = response.json()
        reply = (
            data.get("choices", [{}])[0].get("message", {}).get("content")
            or data.get("choices", [{}])[0].get("content")
            or data.get("choices", [{}])[0].get("text")
            or "No valid response from model."
        )
        reply = reply.strip() if isinstance(reply, str) else str(reply)
        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# 🖼️ IMAGE GENERATION — OpenRouter
# ============================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
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
            "model": "google/gemini-2.5-flash-image-preview",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "modalities": ["image", "text"]
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
        if response.status_code != 200:
            return jsonify({"error": f"OpenRouter image generation failed ({response.status_code})", "details": response.text}), 502

        data = response.json()
        image_url = None
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            if "images" in message and len(message["images"]) > 0:
                image_url = message["images"][0]["image_url"]["url"]

        if image_url:
            return jsonify({"image": image_url}), 200
        else:
            return jsonify({"error": "No image returned by model"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# 🧩 IMAGE MODIFICATION / INPAINTING — OpenRouter
# ============================================================
@app.route("/modify_image", methods=["POST"])
def modify_image():
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
            "model": "google/gemini-2.5-flash-image-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "modalities": ["image", "text"]
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return jsonify({"error": f"OpenRouter image modification failed ({response.status_code})", "details": response.text}), 502

        data = response.json()
        modified_url = None
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            if "images" in message and len(message["images"]) > 0:
                modified_url = message["images"][0]["image_url"]["url"]

        if modified_url:
            return jsonify({"modified_image": modified_url}), 200
        else:
            return jsonify({"error": "No modified image returned"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# 🌐 Root route
# ============================================================
@app.route("/")
def home():
    return jsonify({"status": "OpenRouter unified backend running ✅"}), 200

# ============================================================
# 🚀 Run server
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

