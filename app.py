import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== API KEYS =====
HF_API_KEY = os.getenv("HFACCESKEY")  # Hugging Face API Key
API_KEY = os.getenv("DEEPNAME_KEY", "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu")  # DeepInfra/OpenAI Key

HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_IMAGE_MODIFY_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-refiner-1.0"
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
            image_bytes = response.content

        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(image_bytes)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== IMAGE MODIFICATION (img2img) =====
@app.route("/image_modify", methods=["POST"])
def image_modify():
    """
    Accepts a user-uploaded image and a prompt to modify it.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    prompt = request.form.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # Save uploaded image
    upload_path = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
    file.save(upload_path)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        with open(upload_path, "rb") as img_file:
            response = requests.post(
                HF_IMAGE_MODIFY_URL,
                headers=headers,
                files={"image": img_file},
                data={"inputs": prompt},
                timeout=90
            )

        if response.status_code != 200:
            return jsonify({"error": "Image modification failed", "details": response.text}), response.status_code

        try:
            image_base64 = response.json()[0]["generated_image"]
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            image_bytes = response.content

        filename = f"modified_{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(image_bytes)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"modified_image_url": image_url}), 200

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

