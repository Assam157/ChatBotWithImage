import openai
import requests 
from flask import Flask,request,jsonify
from flask_cors import CORS
import os
app=Flask(__name__)
CORS(app)
 
HF_API_KEY=os.getenv("HFACCESKEY")
API_KEY = "riXezrVqPczSVIcHnsqxlsFkiKFiiyQu"
url = "https://api.deepinfra.com/v1/openai/chat/completions"
 
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
@app.route("/",methods=["GET"])
@app.route("/Image",methods=["POST"])
def Image():
    data = request.json
    prompt = data.get("message", "").strip()

    if not prompt:
        return jsonify({"error": "No Prompt Provided"}), 400

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    response = requests.post(HF_IMAGE_URL, json=payload, headers=headers)

    if response.status_code == 200:
        # Hugging Face API returns raw image bytes
        image_data = response.content
        image_path = f"static/{payload}.png"

        with open(image_path, "wb") as f:
            f.write(image_data)
        
        return jsonify({"image_url": f"http://127.0.0.1:5000/{image_path}"})
        os.remove(image_data)
         
    else:
        return jsonify({"error": "Image generation failed", "details": response.text}), response.status_code
        
@app.route("/chat",methods=["POST"])
 
def chat():
   data=request.json
   prompt=data.get("message","")
    
   headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
   payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # You can change to "meta-llama/Meta-Llama-3-8B"
        "messages": [{"role": "user", "content": prompt}]
    }
    
   response = requests.post(url, json=payload, headers=headers)

   if response.status_code == 200:
        return jsonify({"reply": response.json()["choices"][0]["message"]["content"].strip()})
   else:
        return jsonify({"error": response.json()}), response.status_code

if __name__ == "__main__":
    app.run(debug=True,port="0.0.0.0",use_reloader=False)
