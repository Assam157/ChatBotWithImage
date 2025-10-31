import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)

# ===== ENABLE CORS for your frontend =====
CORS(app, resources={r"/*": {"origins": "*"}})  # ✅ Allow all origins

# ===== API KEYS =====
HF_API_KEY = os.getenv("HFACCESKEY", "")  # Your Hugging Face access token (set in Render dashboard)

# ===== MODEL ENDPOINTS =====
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_IMAGE_MODIFY_URL = "https://api-inference.huggingface.co/models/fal-ai/instruct-pix2pix"
CHAT_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# ===== Serve static files (for saved images) =====
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# ===== Root =====
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask backend running successfully!", 200

# =====================================================
# 🔹 IMAGE GENERATION
# =====================================================
@app.route("/image", methods=["POST", "OPTIONS"])
def image():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_IMAGE_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

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


# =====================================================
# 🔹 IMAGE MODIFICATION (img2img)
# =====================================================
@app.route("/image_modify", methods=["POST", "OPTIONS"])
def image_modify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    prompt = request.form.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "No modification prompt provided"}), 400

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    upload_path = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
    file.save(upload_path)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        with open(upload_path, "rb") as img_file:
            files = {"image": img_file}
            data = {"inputs": prompt}
            response = requests.post(HF_IMAGE_MODIFY_URL, headers=headers, files=files, data=data, timeout=90)

        if response.status_code != 200:
            return jsonify({"error": "Image modification failed", "details": response.text}), response.status_code

        image_bytes = response.content
        filename = f"modified_{uuid.uuid4()}.png"
        path = os.path.join("static", filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"modified_image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# 🔹 CHAT ENDPOINT
# =====================================================
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt
    }

    try:
        response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            return jsonify({"error": "Chat request failed", "details": response.text}), response.status_code

        output = response.json()
        reply = ""
        if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
            reply = output[0]["generated_text"]
        elif isinstance(output, dict) and "generated_text" in output:
            reply = output["generated_text"]
        else:
            reply = "Sorry, no response generated."

        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# 🔹 RUN SERVER
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


