const express = require("express");
const { isPlainObject, handleChoose, handleRepair } = require("./logic");

const app = express();
app.use(express.json());

// Malformed JSON body -> 400 INVALID_INPUT (must come after express.json())
app.use((err, req, res, next) => {
  if (err) {
    return res.status(400).json({ error: "INVALID_INPUT" });
  }
  next();
});

app.post("/adapt", (req, res) => {
  const body = req.body;

  if (!isPlainObject(body) || (body.operation !== "choose" && body.operation !== "repair")) {
    return res.status(400).json({ error: "INVALID_INPUT" });
  }

  const result = body.operation === "choose" ? handleChoose(body) : handleRepair(body);
  return res.status(200).json(result);
});

// Anything else -> 400 INVALID_INPUT (covers unknown routes / wrong method on /adapt)
app.use((req, res) => {
  res.status(400).json({ error: "INVALID_INPUT" });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`adapt-service listening on port ${PORT}`);
});
