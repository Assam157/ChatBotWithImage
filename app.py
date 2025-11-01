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
        reply = None

        try:
            if "choices" in data and len(data["choices"]) > 0:
                # Extract actual assistant content safely
                message_content = data["choices"][0]["message"].get("content", "")
                if isinstance(message_content, list):
                    # Sometimes OpenRouter returns structured content
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
        # Preflight CORS response
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

        # Use OpenRouter's /images/generations endpoint
        response = requests.post(
            "https://openrouter.ai/api/v1/images/generations",
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

        # ✅ Try to extract image URL or base64 data
        image_url = None
        if isinstance(data, dict):
            # Sometimes returned as {"data": [{"url": "..."}, ...]}
            if "data" in data and len(data["data"]) > 0:
                image_url = data["data"][0].get("url")

        if image_url:
            return jsonify({"image_url": image_url}), 200
        else:
            return jsonify({
                "error": "No image returned by model",
                "raw": data
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 IMAGE MODIFICATION — via OpenRouter (Flux Pro)
# ============================================================
@app.route("/modify_image", methods=["POST", "OPTIONS"])
def modify_image():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

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
            "model": "stability-ai/sdxl-inpainting",  # ✅ inpainting model
            "image": image_url,
            "prompt": instruction,
            "mask": None,  # optional mask if you use it later
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/images/edits",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter inpainting failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()
        modified_image = None

        # ✅ Extract modified image URL safely
        if "data" in data and len(data["data"]) > 0:
            modified_image = data["data"][0].get("url")

        if modified_image:
            return jsonify({"modified_image": modified_image}), 200
        else:
            return jsonify({"error": "No modified image returned", "raw": data}), 200

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
