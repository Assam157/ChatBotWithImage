// server.js
import express from "express";
import fetch from "node-fetch";
import bodyParser from "body-parser";
import FormData from "form-data";

const app = express();
app.use(bodyParser.json({ limit: "10mb" }));

// ================= IMAGE GENERATION =====================
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

// ================= IMAGE MODIFICATION (PIX2PIX) =====================
app.post("/modify_image", async (req, res) => {
  try {
    const { prompt, imageUrl } = req.body;
    if (!prompt || !imageUrl)
      return res.status(400).json({ error: "Prompt and imageUrl are required" });

    // Download input image
    const imgResponse = await fetch(imageUrl);
    const imgBuffer = await imgResponse.arrayBuffer();

    const formData = new FormData();
    formData.append("inputs", prompt);
    formData.append("image", Buffer.from(imgBuffer), "input.png");

    // Pix2Pix model for edits
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

// ================= CHAT FUNCTION =====================
app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;
    if (!message) return res.status(400).json({ error: "Message required" });

    // Using DeepInfra's free endpoint — no key needed
    const response = await fetch("https://api.deepinfra.com/v1/openai/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages: [{ role: "user", content: message }]
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return res.status(400).json({
        error: "Chat request failed",
        details: errorText,
      });
    }

    const data = await response.json();
    const reply = data?.choices?.[0]?.message?.content?.trim() || "No reply.";

    res.json({ success: true, reply });
  } catch (error) {
    console.error("Chat error:", error);
    res.status(500).json({
      error: "Internal server error",
      details: error.message,
    });
  }
});

// ================== SERVER START =====================
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`✅ Server running on port ${PORT}`));
