from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os

app = Flask(__name__)
CORS(app)

# ====== CONFIGURATION ======
HF_API_KEY = os.getenv("HFACCESKEY")  # set this in Render environment
if not HF_API_KEY:
    raise ValueError("❌ Missing HF_API_KEY in Render Environment Variables!")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

CHAT_MODEL = "microsoft/DialoGPT-medium"
IMG_GEN_MODEL = "stabilityai/stable-diffusion-2"
IMG_MOD_MODEL = "runwayml/stable-diffusion-inpainting"

# ====== 1️⃣ CHAT ENDPOINT ======
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "Missing 'message'"}), 400

        payload = {"inputs": user_input}
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{CHAT_MODEL}",
            headers=HEADERS, json=payload, timeout=60
        )

        if r.status_code != 200:
            return jsonify({"error": f"HuggingFace chat error: {r.text}"}), r.status_code

        result = r.json()
        reply = result[0]["generated_text"] if isinstance(result, list) else str(result)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== 2️⃣ IMAGE GENERATION ENDPOINT ======
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("message", "")
        if not prompt:
            return jsonify({"error": "Missing 'message'"}), 400

        payload = {"inputs": prompt}
        r = requests.post(
            f"https://api-inference.huggingface.co/models/{IMG_GEN_MODEL}",
            headers=HEADERS, json=payload, timeout=120
        )

        if r.status_code != 200:
            return jsonify({"error": f"HuggingFace image gen error: {r.text}"}), r.status_code

        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== 3️⃣ IMAGE MODIFICATION ENDPOINT ======
@app.route("/modify_image", methods=["POST"])
def modify_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        prompt = data.get("instruction", "")

        if not image_url or not prompt:
            return jsonify({"error": "Missing 'image_url' or 'instruction'"}), 400

        payload = {
            "inputs": {
                "image": image_url,
                "prompt": prompt
            }
        }

        r = requests.post(
            f"https://api-inference.huggingface.co/models/{IMG_MOD_MODEL}",
            headers=HEADERS, json=payload, timeout=120
        )

        if r.status_code != 200:
            return jsonify({"error": f"HuggingFace modify error: {r.text}"}), r.status_code

        return jsonify(r.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== ROOT ROUTE ======
@app.route("/")
def home():
    return jsonify({"status": "Backend is running ✅"})


# ====== RUN APP ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
