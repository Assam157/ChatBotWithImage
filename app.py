import os
import base64
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ======== Enable CORS ========
CORS(app, resources={r"/*": {"origins": "*"}})

# ======== API KEYS ========
HF_API_KEY = os.getenv("HFACCESKEY")  # Hugging Face
API_KEY = os.getenv("DEEPNAME_KEY", "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu")  # DeepInfra/OpenAI

HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ======== ROOT ROUTE ========
@app.route("/", methods=["GET"])
def home():
    return "Backend is running!", 200

# ======== CHAT ROUTE ========
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
        response = requests.post(CHAT_URL, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======== IMAGE ROUTE (Dummy base64 for testing) ========
@app.route("/image", methods=["POST"])
def image():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    # Dummy base64 image (for safe deploy)
    dummy_base64 = base64.b64encode(b"This is a dummy image").decode()
    return jsonify({"image_base64": dummy_base64}), 200

    # ===== Uncomment for real Hugging Face API =====
    """
    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}

    response = requests.post(HF_IMAGE_URL, json=payload, headers=headers, timeout=30)
    if response.status_code == 200:
        try:
            image_base64 = response.json()[0]["generated_image"]
        except Exception:
            image_base64 = base64.b64encode(response.content).decode()
        return jsonify({"image_base64": image_base64})
    else:
        return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code
    """

# ======== RUN APP ========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamic port
    app.run(debug=True, host="0.0.0.0", port=port)


