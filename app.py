import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== API KEYS =====
HF_API_KEY = os.getenv("HFACCESKEY") 

# ===== Updated Hugging Face Router Endpoints =====
HF_IMAGE_URL = "https://router.huggingface.co/hf-inference/stabilityai/stable-diffusion-xl-base-1.0"
HF_IMAGE_MODIFY_URL = "https://router.huggingface.co/hf-inference/timbrooks/instruct-pix2pix"
HF_CHAT_URL = "https://router.huggingface.co/hf-inference/mistralai/Mixtral-8x7B-Instruct-v0.1"

# ===== Serve static folder =====
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# ===== Root =====
@app.route("/", methods=["GET"])
def home():
    return "✅ Backend running with Hugging Face Router API", 200
    print("\n=== Registered Flask Routes ===")
    for rule in app.url_map.iter_rules():
        print(rule)
        print("===============================\n")



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
        response = requests.post(HF_IMAGE_URL, json=payload, headers=headers, timeout=90)
        if response.status_code != 200:
            return jsonify({
                "error": "Image generation failed",
                "details": response.text
            }), response.status_code

        # Try to extract image data (base64 or raw bytes)
        try:
            result = response.json()
            if isinstance(result, list) and "generated_image" in result[0]:
                image_base64 = result[0]["generated_image"]
                image_bytes = base64.b64decode(image_base64)
            else:
                image_bytes = response.content
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


# ===== IMAGE MODIFICATION =====
@app.route("/image_modify", methods=["POST"])
def image_modify():
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

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    try:
        with open(upload_path, "rb") as img_file:
            response = requests.post(
                HF_IMAGE_MODIFY_URL,
                headers=headers,
                data={"inputs": prompt},
                files={"image": img_file},
                timeout=90
            )

        if response.status_code != 200:
            return jsonify({
                "error": "Image modification failed",
                "details": response.text
            }), response.status_code

        try:
            result = response.json()
            if isinstance(result, list) and "generated_image" in result[0]:
                image_base64 = result[0]["generated_image"]
                image_bytes = base64.b64decode(image_base64)
            else:
                image_bytes = response.content
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


# ===== CHAT (via Hugging Face Router) =====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_CHAT_URL, json=payload, headers=headers, timeout=45)
        if response.status_code != 200:
            return jsonify({
                "error": "Chat failed",
                "details": response.text
            }), response.status_code

        result = response.json()
        reply = result.get("generated_text") or result[0].get("generated_text", "No response received.")
        return jsonify({"reply": reply.strip()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== RUN APP =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
