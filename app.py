import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== API CONFIG =====
HF_API_KEY = os.getenv("HF_API_KEY")  # optional for free models
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HF_IMAGE_MODIFY_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPNAME_KEY = os.getenv("DEEPNAME_KEY", "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu")

# ===== STATIC FILE SERVING =====
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# ===== ROOT =====
@app.route("/", methods=["GET"])
def home():
    return "✅ Backend running — Stable Diffusion 2 Image Modify + Chat Ready", 200


# ===== IMAGE GENERATION =====
@app.route("/image", methods=["POST"])
def image():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_IMAGE_URL, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            return jsonify({
                "error": "Image generation failed",
                "details": response.text
            }), response.status_code

        try:
            data = response.json()
            image_base64 = data[0]["image"] if isinstance(data, list) else data.get("image")
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
    Modify an uploaded image using Stable Diffusion 2 and a text prompt.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    prompt = request.form.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    upload_path = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
    file.save(upload_path)

    with open(upload_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    payload = {
        "inputs": prompt,
        "image": image_b64,
        "parameters": {
            "guidance_scale": 7.5,
            "num_inference_steps": 40
        }
    }

    try:
        response = requests.post(HF_IMAGE_MODIFY_URL, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            return jsonify({
                "error": "Image modification failed",
                "details": response.text
            }), response.status_code

        try:
            data = response.json()
            image_base64 = data[0]["image"] if isinstance(data, list) else data.get("image")
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            image_bytes = response.content  # fallback

        filename = f"modified_{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(image_bytes)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"modified_image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== CHAT ENDPOINT =====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    headers = {"Authorization": f"Bearer {DEEPNAME_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"].strip()
        return jsonify({"reply": reply}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== RUN SERVER =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
