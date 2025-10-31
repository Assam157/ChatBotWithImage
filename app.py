import os
import uuid
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# =====================================================
# ✅ SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

# === API KEYS ===
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "your_stability_api_key_here")
DEEPNAME_KEY = os.getenv("DEEPNAME_KEY", "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu")

# === ENDPOINTS ===
STABILITY_GEN_URL = "https://api.stability.ai/v2beta/stable-image/generate/core"
STABILITY_EDIT_URL = "https://api.stability.ai/v2beta/stable-image/edit"
CHAT_URL = "https://api.deepinfra.com/v1/openai/chat/completions"


# =====================================================
# ✅ STATIC SERVING
# =====================================================
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/", methods=["GET"])
def home():
    return "✅ Backend running — StabilityAI + Chat + Img2Img ready!", 200


# =====================================================
# 🧩 IMAGE GENERATION
# =====================================================
@app.route("/image", methods=["POST"])
def generate_image():
    """Generate a new image from a text prompt."""
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    files = {"none": (None, prompt)}

    response = requests.post(STABILITY_GEN_URL, headers=headers, files=files, timeout=90)

    if response.status_code != 200:
        return jsonify({
            "error": "Image generation failed",
            "details": response.text
        }), response.status_code

    os.makedirs("static", exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    path = os.path.join("static", filename)
    with open(path, "wb") as f:
        f.write(response.content)

    image_url = f"/static/{filename}"
    return jsonify({"image_url": image_url}), 200


# =====================================================
# ✏️ IMAGE MODIFICATION
# =====================================================
@app.route("/image_modify", methods=["POST"])
def modify_image():
    """Edit an existing image based on a prompt."""
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

    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}"}
    files = {
        "image": open(upload_path, "rb"),
        "prompt": (None, prompt)
    }

    response = requests.post(STABILITY_EDIT_URL, headers=headers, files=files, timeout=90)

    if response.status_code != 200:
        return jsonify({
            "error": "Image modification failed",
            "details": response.text
        }), response.status_code

    filename = f"modified_{uuid.uuid4()}.png"
    path = os.path.join("static", filename)
    with open(path, "wb") as f:
        f.write(response.content)

    image_url = f"/static/{filename}"
    return jsonify({"modified_image_url": image_url}), 200


# =====================================================
# 💬 CHAT ENDPOINT
# =====================================================
@app.route("/chat", methods=["POST"])
def chat():
    """Chat completion using DeepInfra Mixtral model."""
    data = request.json
    prompt = data.get("message", "").strip() if data else ""
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    headers = {
        "Authorization": f"Bearer {DEEPNAME_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(CHAT_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        return jsonify({
            "error": "Chat request failed",
            "details": response.text
        }), response.status_code

    reply = response.json()["choices"][0]["message"]["content"].strip()
    return jsonify({"reply": reply}), 200


# =====================================================
# 🚀 RUN
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
