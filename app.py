from flask import Flask, request, jsonify
from flask_cors import CORS
import os, requests

app = Flask(__name__)
CORS(app)

# === Environment Keys ===
OPENROUTER_KEY = os.getenv("OPENAIKEY")
FALAI_KEY = os.getenv("FALAIKEY")
HF_KEY = os.getenv("HF_KEY")

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
        data = request.get_json(force=True)
        user_input = data.get("message", "").strip()
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
        if response.status_code != 200:
            return jsonify({
                "error": f"OpenRouter request failed ({response.status_code})",
                "details": response.text
            }), 502

        data = response.json()
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "No valid response.")
        return jsonify({"reply": reply}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🖼️ IMAGE GENERATION (Fal.AI → fallback HuggingFace)
# ============================================================
@app.route("/generate_image", methods=["POST"])
def generate_image():
    try:
        prompt = request.json.get("message", "").strip()
        if not prompt:
            return jsonify({"error": "Missing prompt"}), 400

        # --- Try FalAI ---
        if FALAI_KEY:
            headers = {"Authorization": f"Key {FALAI_KEY}", "Content-Type": "application/json"}
            payload = {"prompt": prompt, "num_images": 1}

            fal_response = requests.post(FAL_IMAGE_URL, headers=headers, json=payload, timeout=90)
            if fal_response.ok:
                result = fal_response.json()
                if "images" in result:
                    return jsonify({"image": result["images"][0]["url"]}), 200

                elif "error" in result:
                    print("FalAI Error:", result["error"])

        # --- Fallback to HuggingFace ---
        if HF_KEY:
            hf_headers = {"Authorization": f"Bearer {HF_KEY}"}
            hf_payload = {"inputs": prompt}
            hf_response = requests.post(HF_IMAGE_URL, headers=hf_headers, json=hf_payload, timeout=90)

            try:
                hf_json = hf_response.json()
                if isinstance(hf_json, dict) and "error" in hf_json:
                    return jsonify({"error": "HF API error", "details": hf_json}), 502
                return jsonify({"image": hf_json}), 200
            except Exception:
                return jsonify({"error": "Invalid HuggingFace response"}), 502

        return jsonify({"error": "All providers failed or keys missing"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🧩 IMAGE MODIFICATION / INPAINTING
# ============================================================
@app.route("/modify_image", methods=["POST"])
def modify_image():
    try:
        data = request.get_json(force=True)
        image_url = data.get("image_url", "").strip()
        instruction = data.get("instruction", "").strip()

        if not image_url or not instruction:
            return jsonify({"error": "Missing fields (image_url/instruction)"}), 400

        # --- Try FalAI ---
        if FALAI_KEY:
            headers = {"Authorization": f"Key {FALAI_KEY}", "Content-Type": "application/json"}
            payload = {"image_url": image_url, "prompt": instruction}
            fal_response = requests.post(FAL_INPAINT_URL, headers=headers, json=payload, timeout=90)

            if fal_response.ok:
                result = fal_response.json()
                if "images" in result:
                    return jsonify({"modified_image": result["images"][0]["url"]}), 200
                elif "error" in result:
                    print("FalAI inpaint error:", result["error"])

        # --- Fallback to HuggingFace ---
        if HF_KEY:
            hf_headers = {"Authorization": f"Bearer {HF_KEY}"}
            hf_payload = {"inputs": {"image": image_url, "prompt": instruction}}
            hf_response = requests.post(HF_INPAINT_URL, headers=hf_headers, json=hf_payload, timeout=90)

            try:
                hf_json = hf_response.json()
                if isinstance(hf_json, dict) and "error" in hf_json:
                    return jsonify({"error": "HF Inpaint failed", "details": hf_json}), 502
                return jsonify({"modified_image": hf_json}), 200
            except Exception:
                return jsonify({"error": "Invalid HuggingFace inpaint response"}), 502

        return jsonify({"error": "All providers failed or keys missing"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 🌐 Root route
# ============================================================
@app.route("/")
def home():
    return jsonify({"status": "Unified AI backend running ✅"}), 200


# ============================================================
# 🚀 Run server
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
