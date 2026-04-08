require("dotenv").config();
const express = require("express");
const cors = require("cors");
const analyzeRoute = require("./routes/analyze");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// Routes
app.use("/api/analyze", analyzeRoute);

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "agrisense-ai-server" });
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
