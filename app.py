import os
import uuid
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ===== API KEYS (from GitHub Secrets / Environment Variables) =====
HF_API_KEY = os.getenv("HF_KEY")
O_R_KEY = os.getenv("OR_KEY")

# ===== Endpoints =====
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_IMAGE_MODIFY_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

# ===== Serve static folder =====
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)

# ===== Root =====
@app.route("/", methods=["GET"])
def home():
    return "✅ Backend running with Hugging Face & OpenRouter integration", 200


# ===== IMAGE GENERATION =====
@app.route("/image", methods=["POST"])
def image():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    if not HF_API_KEY:
        return jsonify({"error": "Missing Hugging Face API key"}), 500

    headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
    payload = {"inputs": prompt}

    try:
        response = requests.post(HF_IMAGE_URL, json=payload, headers=headers, timeout=90)

        if response.status_code != 200:
            return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code

        image_bytes = response.content
        os.makedirs("static", exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join("static", filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        # On GitHub/Render, use dynamic base URL
        base_url = request.host_url.rstrip("/")
        image_url = f"{base_url}/static/{filename}"
        return jsonify({"image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== IMAGE MODIFICATION =====
@app.route("/image_modify", methods=["POST"])
def image_modify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    prompt = request.form.get("message", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    if not HF_API_KEY:
        return jsonify({"error": "Missing Hugging Face API key"}), 500

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    upload_path = os.path.join("uploads", f"{uuid.uuid4()}_{file.filename}")
    file.save(upload_path)

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    with open(upload_path, "rb") as img_file:
        image_bytes = img_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {"inputs": prompt, "image": image_b64}

    try:
        response = requests.post(HF_IMAGE_MODIFY_URL, headers=headers, json=payload, timeout=90)

        if response.status_code != 200:
            return jsonify({"error": "Image modification failed", "details": response.text}), response.status_code

        image_bytes = response.content
        filename = f"modified_{uuid.uuid4()}.png"
        path = os.path.join("static", filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        base_url = request.host_url.rstrip("/")
        image_url = f"{base_url}/static/{filename}"
        return jsonify({"modified_image_url": image_url}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== CHAT (via OpenRouter) =====
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    if not O_R_KEY:
        return jsonify({"error": "Missing OpenRouter API key"}), 500

    # 💪 Using Llama 3.1 70B for high-quality responses
    model = "meta-llama/llama-3.1-70b-instruct"

    headers = {
        "Authorization": f"Bearer {O_R_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": request.host_url,
        "X-Title": "GitHub-Deployed Chatbot"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, creative assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            return jsonify({"error": "Chat failed", "details": response.text, "model_used": model}), response.status_code

        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        return jsonify({"reply": reply.strip(), "model_used": model}), 200

    except Exception as e:
        return jsonify({"error": str(e), "model_used": model}), 500


# ===== RUN APP =====
if __name__ == "__main__":
    print("\n✅ Flask backend is starting...")
    print("Available routes:\n")
    with app.test_request_context():
        for rule in app.url_map.iter_rules():
            print(rule)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
