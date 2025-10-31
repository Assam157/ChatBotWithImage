import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ================== API URLs ==================
IMAGE_GEN_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
IMAGE_MODIFY_URL = "https://api-inference.huggingface.co/models/timbrooks/instruct-pix2pix"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# ================== STATIC SERVE ==================
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/", methods=["GET"])
def home():
    return "✅ Flask Backend Running — Image + Modify + Chat Active!", 200


# ================== IMAGE GENERATION ==================
@app.route("/image", methods=["POST"])
def generate_image():
    try:
        data = request.json
        prompt = data.get("message", "").strip() if data else ""
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        payload = {"inputs": prompt}
        headers = {"Accept": "application/json"}  # Important for Hugging Face

        response = requests.post(IMAGE_GEN_URL, headers=headers, json=payload, timeout=90)
        if not response.ok:
            return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

        # Save image
        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(response.content)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================== IMAGE MODIFICATION ==================
@app.route("/image_modify", methods=["POST"])
def image_modify():
    try:
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

        # Prepare multipart form-data
        with open(upload_path, "rb") as img_file:
            files = {"image": img_file}
            data = {"inputs": prompt}
            headers = {"Accept": "application/json"}

            response = requests.post(IMAGE_MODIFY_URL, headers=headers, files=files, data=data, timeout=90)

        if not response.ok:
            return jsonify({"error": "Image modification failed", "details": response.text}), response.status_code

        filename = f"modified_{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(response.content)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"modified_image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================== CHAT COMPLETIONS ==================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        message = data.get("message", "").strip() if data else ""
        if not message:
            return jsonify({"error": "No message provided"}), 400

        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "messages": [{"role": "user", "content": message}]
        }

        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
        if not response.ok:
            return jsonify({"error": "Chat API failed", "details": response.text}), response.status_code

        data = response.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        return jsonify({"reply": reply or "No reply"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ================== RUN SERVER ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

