import json
import matplotlib.pyplot as plt

# Load the data
with open("experiment_results.json", "r") as f:
    data = json.load(f)

# Create the plot 
plt.figure(figsize=(10, 6))

# Plot the three baselines
plt.plot(data["batches"], data["linucb_only"], label="LinUCB Baseline", color="green", linestyle="-")
plt.plot(data["batches"], data["llm_only"], label="LLM-Based Agent", color="purple", linestyle="--")
plt.plot(data["batches"], data["hybrid"], label="Our Method (Hybrid)", color="red", linestyle="-", linewidth=2)

# Formatting to match the Chen et al. paper
plt.title("Average Cumulative Reward over Time", fontsize=16)
plt.xlabel("Iteration (Batches)", fontsize=14)
plt.ylabel("Average Cumulative Reward", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)

# Save and show
plt.savefig("reward_multiplication.png", dpi=300)
plt.show()
