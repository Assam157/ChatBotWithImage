from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)
CORS(app)

# === Environment Keys ===
OPENROUTER_KEY = os.getenv("OPENAIKEY")
FALAI_KEY = os.getenv("FALAIKEY")
HF_KEY = os.getenv("HFACCESKEY")

# === API URLs ===
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
FAL_IMAGE_URL = "https://fal.run/fal-ai/flux-pro"
FAL_INPAINT_URL = "https://fal.run/fal-ai/flux-inpaint"
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
HF_INPAINT_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-inpainting"

# ============================================================
# 🧠 CHAT (OpenRouter)
# ============================================================
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        if not user_input:
            return jsonify({"error": "Missing message"}), 400

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ]
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            reply = data["choices"][0]["message"]["content"]
        else:
            reply = "No valid response."

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🖼️ IMAGE GENERATION (Fal.AI → fallback HuggingFace)
# ============================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        prompt = request.json.get("message", "")
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        headers = {"Authorization": f"Key {FALAI_KEY}", "Content-Type": "application/json"}
        payload = {"prompt": prompt, "num_images": 1}

        # Try Fal.AI first
        fal_response = requests.post(FAL_IMAGE_URL, headers=headers, json=payload, timeout=90)
        if fal_response.status_code == 200:
            result = fal_response.json()
            if "images" in result:
                return jsonify({"image": result["images"][0]["url"]})
        else:
            # Fallback to HuggingFace
            hf_headers = {"Authorization": f"Bearer {HF_KEY}"}
            hf_payload = {"inputs": prompt}
            hf_response = requests.post(HF_IMAGE_URL, headers=hf_headers, json=hf_payload)
            return jsonify({"image": hf_response.json()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 IMAGE MODIFICATION / INPAINTING
# ============================================================
@app.route("/modify_image", methods=["POST"])
def modify_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url", "")
        instruction = data.get("instruction", "")
        if not image_url or not instruction:
            return jsonify({"error": "Missing fields"}), 400

        headers = {"Authorization": f"Key {FALAI_KEY}", "Content-Type": "application/json"}
        payload = {"image_url": image_url, "prompt": instruction}

        fal_response = requests.post(FAL_INPAINT_URL, headers=headers, json=payload, timeout=90)

        if fal_response.status_code == 200:
            result = fal_response.json()
            if "images" in result:
                return jsonify({"modified_image": result["images"][0]["url"]})
        else:
            # Fallback to HuggingFace
            hf_headers = {"Authorization": f"Bearer {HF_KEY}"}
            hf_payload = {"inputs": {"image": image_url, "prompt": instruction}}
            hf_response = requests.post(HF_INPAINT_URL, headers=hf_headers, json=hf_payload)
            return jsonify({"modified_image": hf_response.json()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return jsonify({"status": "Unified AI backend running ✅"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
