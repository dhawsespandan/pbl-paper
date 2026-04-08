const express = require("express");
const router = express.Router();
const axios = require("axios");
const FormData = require("form-data");
const upload = require("../middleware/upload");

router.post("/", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image provided" });
    }

    // Forward image to Python FastAPI service
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    const pythonRes = await axios.post(
      `${process.env.PYTHON_SERVICE_URL}/predict`,
      form,
      { headers: form.getHeaders() }
    );

    return res.json(pythonRes.data);
  } catch (err) {
    if (err.response) {
      const { status, data } = err.response;
      console.error("Analysis error (upstream):", status, data);
      if (data && typeof data === "object") {
        return res.status(status).json(data);
      }
      return res.status(status).json({
        error: typeof data === "string" ? data : "Analysis failed",
      });
    }
    console.error("Analysis error:", err.message);
    const code = err.code === "ECONNREFUSED" ? 503 : 502;
    return res.status(code).json({
      error: "Model service unavailable",
      detail: err.message,
    });
  }
});

module.exports = router;