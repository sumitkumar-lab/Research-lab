import json
import matplotlib.pyplot as plt

with open("training_outputs/metrics.json") as f:
    metrics = json.load(f)


plt.figure(figsize=(18, 12))

steps = metrics["step"]
# ---------------- Loss ----------------
plt.subplot(2, 3, 1)
plt.plot(steps, metrics["train_loss"], label="Train Loss")
plt.plot(steps, metrics["val_loss"], label="Val Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()

# ---------------- Accuracy ----------------
plt.subplot(2, 3, 2)
plt.plot(steps, metrics["val_accuracy"])
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")

# ---------------- Perplexity ----------------
plt.subplot(2, 3, 3)
plt.plot(steps, metrics["val_perplexity"])
plt.xlabel("Steps")
plt.ylabel("Perplexity")
plt.title("Validation Perplexity")

# ---------------- Learning Rate ----------------
plt.subplot(2, 3, 4)
plt.plot(steps, metrics["learning_rate"])
plt.xlabel("Steps")
plt.ylabel("LR")
plt.title("Learning Rate Schedule")

# ---------------- Throughput ----------------
plt.subplot(2, 3, 5)
plt.plot(steps, metrics["tokens_per_sec"])
plt.xlabel("Steps")
plt.ylabel("Tokens/sec")
plt.title("Training Throughput")

# ---------------- Time ----------------
plt.subplot(2, 3, 6)
plt.plot(steps, metrics["elapsed_time"])
plt.xlabel("Steps")
plt.ylabel("Seconds")
plt.title("Elapsed Training Time")

plt.show()