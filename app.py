# ============================================================
# 🧠 Hugging Face Inference Backend (Router API, 2025 update)
# ============================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)

# --- Allow all origins + preflight ---
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)

HF_KEY = os.getenv("HF_KEY")  # <-- set in your environment
if not HF_KEY:
    print("⚠️  Warning: HF_KEY not found in environment.")

BASE_URL = "https://router.huggingface.co/hf-inference"

# ============================================================
# 🎨 TEXT → IMAGE GENERATION
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
            "Authorization": f"Bearer {HF_KEY}",
            "Content-Type": "application/json"
        }

        # You can change model below to another free provider model
        payload = {
            "inputs": prompt,
            "parameters": {"width": 512, "height": 512},
            "model": "black-forest-labs/FLUX.1-dev"
        }

        response = requests.post(
            f"{BASE_URL}/models/black-forest-labs/FLUX.1-dev",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                "error": f"HF image generation failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()

        # HF returns base64 or URLs depending on the provider
        if isinstance(data, dict) and "images" in data:
            image_data = data["images"][0]
            return jsonify({"image_base64": image_data}), 200

        return jsonify({"error": "No image returned", "raw": data}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 IMAGE MODIFICATION / INPAINTING
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
            "Authorization": f"Bearer {HF_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": {
                "image": image_url,
                "prompt": instruction
            },
            "model": "timbrooks/instruct-pix2pix"
        }

        response = requests.post(
            f"{BASE_URL}/models/timbrooks/instruct-pix2pix",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return jsonify({
                "error": f"HF image modification failed ({response.status_code})",
                "details": response.text
            }), response.status_code

        data = response.json()

        if isinstance(data, dict) and "images" in data:
            image_data = data["images"][0]
            return jsonify({"modified_image": image_data}), 200

        return jsonify({"error": "No modified image returned", "raw": data}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🌐 Root Endpoint
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "✅ Hugging Face Inference backend running"}), 200


# ============================================================
# 🚀 Run Server
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
