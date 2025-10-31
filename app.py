import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# === Hugging Face Model URL ===
HF_API_KEY = os.getenv("HF_API_KEY")  # optional; model is public
HF_IMAGE_MODIFY_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"

# === Static folder for serving modified images ===
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

@app.route("/", methods=["GET"])
def home():
    return "✅ Image modification backend running!", 200


# === Image Modification Endpoint ===
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

    # Save uploaded image
    upload_path = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
    file.save(upload_path)

    # Prepare request
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    with open(upload_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": prompt,
        "image": image_b64,
        "parameters": {"guidance_scale": 7.5, "num_inference_steps": 30},
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
            image_bytes = response.content  # fallback if Hugging Face returns raw image bytes

        # Save modified image
        filename = f"modified_{uuid.uuid4()}.png"
        path = os.path.join("static", filename)
        with open(path, "wb") as f:
            f.write(image_bytes)

        image_url = f"https://chatbotwithimagebackend.onrender.com/static/{filename}"
        return jsonify({"modified_image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
