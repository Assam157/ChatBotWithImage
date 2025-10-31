from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)

# ✅ allow all origins and all methods (fixes your CORS preflight)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

HF_API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_TOKEN = bos.getenv("HFACESSKEY")

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Backend live"})


# =========================
# 🗨️ Chat endpoint
# =========================
@app.route("/chat", methods=["OPTIONS", "POST"])
def chat():
    # handle preflight
    if request.method == "OPTIONS":
        return _cors_preflight_response()
    try:
        data = request.get_json()
        prompt = data.get("message", "")

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200}
        }

        resp = requests.post(HF_API_URL, headers=headers, json=payload)
        if resp.status_code != 200:
            return jsonify({"error": "HF request failed", "details": resp.text}), resp.status_code

        reply_text = resp.json()[0]["generated_text"]
        return _corsify_actual_response(jsonify({"reply": reply_text}))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🖼️ Image Modify endpoint
# =========================
@app.route("/image_modify", methods=["OPTIONS", "POST"])
def image_modify():
    if request.method == "OPTIONS":
        return _cors_preflight_response()

    try:
        if "image" not in request.files or "prompt" not in request.form:
            return jsonify({"error": "Missing image or prompt"}), 400

        image_file = request.files["image"]
        prompt = request.form["prompt"]

        files = {"image": (image_file.filename, image_file.read(), image_file.content_type)}
        data = {"inputs": prompt}

        r = requests.post(
            "https://api-inference.huggingface.co/models/fal-ai/instruct-pix2pix",
            headers=headers,
            data=data,
            files=files,
        )

        return _corsify_actual_response(jsonify(r.json()))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- CORS helpers ----------
def _cors_preflight_response():
    response = jsonify({"status": "ok"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response, 200

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run(debug=True)
