from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)

# === 🌍 Allow all CORS (with OPTIONS pass) ===
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)

OPENROUTER_KEY = os.getenv("OPENAIKEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# 🧠 CHAT — via OpenRouter
# ============================================================
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

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
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input}
            ]
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter request failed ({response.status_code})",
                "details": response.text
            }), 502

        data = response.json()
        reply = ""

        try:
            if "choices" in data and len(data["choices"]) > 0:
                message_content = data["choices"][0]["message"].get("content", "")
                if isinstance(message_content, list):
                    reply = "".join([c.get("text", "") for c in message_content if isinstance(c, dict)])
                else:
                    reply = message_content
        except Exception:
            reply = "Error extracting reply."

        if not reply:
            reply = "⚠️ No valid response from AI."

        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🖼️ IMAGE GENERATION — via OpenRouter
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
            "model": "stabilityai/stable-diffusion-3",
            "messages": [
                {"role": "user", "content": f"Generate an image of: {prompt}"}
            ]
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter image generation failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()
        image_url = None

        # ✅ Extract image URL from multimodal response
        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0]["message"]
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        break

        if image_url:
            return jsonify({"image_url": image_url}), 200
        else:
            return jsonify({
                "error": "No image returned by model",
                "raw": data
            }), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 IMAGE MODIFICATION — via OpenRouter (Refiner)
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
            "model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "messages": [
                {"role": "user", "content": f"Modify the image {image_url} as follows: {instruction}"}
            ]
        }

        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter inpaint failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()
        modified_image = None

        if "choices" in data and len(data["choices"]) > 0:
            msg = data["choices"][0]["message"]
            if "content" in msg and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "image_url":
                        modified_image = item["image_url"]["url"]
                        break

        if modified_image:
            return jsonify({"modified_image": modified_image}), 200
        else:
            return jsonify({
                "error": "No image returned by model",
                "raw": data
            }), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🌐 Root Endpoint
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Unified OpenRouter AI backend running ✅"}), 200


# ============================================================
# 🚀 Run App
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
