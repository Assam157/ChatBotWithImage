// server.js
import express from "express";
import fetch from "node-fetch";
import bodyParser from "body-parser";
import FormData from "form-data";

const app = express();
app.use(bodyParser.json());

// =============== IMAGE GENERATION ===================
// Example models: 
// "stabilityai/stable-diffusion-2"
// "runwayml/stable-diffusion-v1-5"
// "prompthero/openjourney"
app.post("/generate_image", async (req, res) => {
  try {
    const { prompt } = req.body;
    if (!prompt) return res.status(400).json({ error: "Prompt is required" });

    const response = await fetch(
      "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ inputs: prompt }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(400).json({
        error: "Image generation failed",
        details: errorText,
      });
    }

    // Get image bytes → base64 encode
    const arrayBuffer = await response.arrayBuffer();
    const base64Image = Buffer.from(arrayBuffer).toString("base64");

    res.json({
      success: true,
      image: `data:image/png;base64,${base64Image}`,
    });
  } catch (error) {
    console.error("Generation error:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message,
    });
  }
});

// =============== IMAGE MODIFICATION (Pix2Pix) ===================
app.post("/modify_image", async (req, res) => {
  try {
    const { prompt, imageUrl } = req.body;
    if (!prompt || !imageUrl)
      return res.status(400).json({ error: "Prompt and imageUrl are required" });

    // Fetch input image and prepare form data
    const imgResponse = await fetch(imageUrl);
    const imgBuffer = await imgResponse.arrayBuffer();

    const formData = new FormData();
    formData.append("inputs", prompt);
    formData.append("image", Buffer.from(imgBuffer), "input.png");

    // Use instruct-pix2pix for modification
    const response = await fetch(
      "https://api-inference.huggingface.co/models/timbrooks/instruct-pix2pix",
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(400).json({
        error: "Image modification failed",
        details: errorText,
      });
    }

    const arrayBuffer = await response.arrayBuffer();
    const base64Image = Buffer.from(arrayBuffer).toString("base64");

    res.json({
      success: true,
      image: `data:image/png;base64,${base64Image}`,
    });
  } catch (error) {
    console.error("Modification error:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message,
    });
  }
});

// =============== SERVER START ===================
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
