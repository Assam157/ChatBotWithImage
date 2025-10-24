import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HF_API_KEY = os.getenv("HFACCESKEY")  # Hugging Face
API_KEY = "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu"  # DeepInfra/OpenAI

HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ======== SERVE STATIC FILES ========
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# ======== IMAGE GENERATION ========
@app.route("/Image", methods=["POST"])
def Image():
    data = request.json
    prompt = data.get("message", "").strip()
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}
    response = requests.post(HF_IMAGE_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            image_base64 = response.json()[0]["generated_image"]
            image_data = base64.b64decode(image_base64)
        except Exception:
            image_data = response.content

        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        image_path = os.path.join("static", filename)
        with open(image_path, "wb") as f:
            f.write(image_data)

        return jsonify({"image_url": f"https://chatbotwithimagebackend.onrender.com/static/{filename}"})
    else:
        return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

# ======== CHAT ========
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip()
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(CHAT_URL, json=payload, headers=headers)
    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply})
    else:
        return jsonify({"error": response.json()}), response.status_code

# ======== RUN APP ========
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
