import os
import uuid
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== API KEYS =====
HF_API_KEY = os.getenv("HFACCESKEY")  # Hugging Face API Key
API_KEY = os.getenv("DEEPNAME_KEY", "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu")  # DeepInfra/OpenAI Key

HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ===== Serve static folder for saved images =====
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# ===== ROOT =====
@app.route("/", methods=["GET"])
def home():
    return "Backend running!", 200

# ===== IMAGE GENERATION =====
@app.route("/image", methods=["POST"])
def image():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_IMAGE_URL, json=payload, headers=headers, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

        # Hugging Face can return JSON with base64 or raw bytes
        try:
            image_base64 = response.json()[0]["generated_image"]
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            # fallback: raw bytes
            image_bytes = response.content

        # Save image to static folder
        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(image_bytes)

        # Return URL
        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== CHAT =====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(CHAT_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== RUN APP =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

