# LinkedIn Showcase Post — AI Security & Federated Learning Project

Below are **two ready-to-post options** for LinkedIn (a detailed engineering storytelling post and a shorter, high-impact summary version), plus recommended images to attach.

---

## Option 1: Detailed Engineering Post (Recommended for AI/ML Recruiters)

How easy is it to fool a production deep learning model or poison a distributed Federated Learning system—and how do we actually defend against it? 🛡️🤖

Over the past few weeks, Subhi Alloush and I built an end-to-end **AI Security & Adversarial Robustness** benchmark at Blekinge Institute of Technology (BTH), investigating both centralized evasion attacks and decentralized poisoning in Federated Learning.

Here is what we built and what the data revealed:

🔹 **Part 1: Targeted Evasion Attacks & Multi-Layer Input Defense (ResNet-50)**
Using the Adversarial Robustness Toolbox (ART), we executed a targeted **Projected Gradient Descent (PGD)** white-box attack against a pretrained ImageNet ResNet-50 classifier.
• By modifying pixel values by at most ±2.5 (imperceptible to the human eye), we forced the model to misclassify a **Koala (99.99% confidence)** as a **Tractor (99.9% confidence)**.
• **The Defense:** Because retraining a foundation/ImageNet-scale model via Adversarial Training is computationally expensive, we engineered a lightweight **Input Preprocessing Defense** combining **Feature Squeezing (4-bit depth reduction)** and **JPEG Compression (quality 60)**.
• **Result:** The defense stripped out the high-frequency adversarial perturbations before the first convolutional layer—restoring **94.9% accuracy** on the attacked image and blocking adaptive PGD attacks directly targeting the defended pipeline, with zero accuracy loss on clean images.

🔹 **Part 2: Federated Learning Data Poisoning & Byzantine-Robust Aggregation (Flower + PyTorch)**
We simulated a 5-client Federated Learning network on CIFAR-10 over 35 communication rounds using **Flower (`flwr`)** and **Ray**, where malicious clients launched **systematic label-flipping data poisoning attacks** (20% and 40% compromised clients).
• **Standard `FedAvg` Vulnerability:** With 1 attacker (20%), global accuracy dropped from **73.59% to 64.49%**. With 2 attackers (40%), the global model collapsed to **44.76%** (Cohen's Kappa: **0.3862**).
• **Why Server Momentum (`FedAvgM`) Fails as a Defense:** Testing `FedAvgM` ($\beta = 0.9$) showed that momentum actually *amplifies* poisoning (dropping to **49.32%** with 1 attacker), because the server's velocity vector accumulates and carries malicious gradients across future rounds.
• **Byzantine-Robust Defenses (`FedMedian` vs. `FedTrimmedAvg`):**
  - Under **1 Attacker (20% poisoned)**, **Coordinate-wise Trimmed Mean (`FedTrimmedAvg`, $\beta=0.2$)** performed best (**72.32%**), slightly outperforming **`FedMedian` (72.23%)** by discarding the outlier tails and averaging the remaining 3 honest clients.
  - Under **2 Attackers (40% poisoned)**, **`FedMedian` (59.67%)** significantly outperformed **`FedTrimmedAvg` (52.77%)**, proving the real-world importance of `FedMedian`'s theoretical **50% breakdown point** when the attacker ratio exceeds the trim threshold.

💡 **Key Takeaway for Real-World AI Systems:**
Whether deploying centralized vision models in autonomous systems or training privacy-preserving Federated Learning models across hospital or edge devices, standard optimization (`FedAvg` / `FedAvgM`) assumes honest participants. Building trustworthy AI requires combining input sanitization at inference time with statistically robust aggregation at training time.

🔗 **Full code, Jupyter Notebooks, and benchmark plots on GitHub:**
https://github.com/boostx1/Security-in-AI-systems

#AISecurity #MachineLearning #FederatedLearning #DeepLearning #AdversarialML #PyTorch #FlowerFL #CyberSecurity #AIEngineering #BTH

---

## Option 2: Short & Punchy Version

What happens when 40% of clients in a Federated Learning network actively poison the global model—or when an invisible ±2.5 pixel perturbation fools ResNet-50? 🛡️🧠

Subhi Alloush and I just wrapped up a hands-on **AI Security** engineering project exploring attacks and defenses across both centralized and decentralized ML systems:

1️⃣ **Centralized Vision Security (ResNet-50 + ART):**
• Executed a targeted **PGD white-box evasion attack** that turned a 100% confident *Koala* prediction into a 100% confident *Tractor* without visible changes.
• Built a zero-retraining **Feature Squeezing (4-bit) + JPEG Compression** preprocessing pipeline that neutralized the attack and restored **94.9% confidence** on the true class.

2️⃣ **Federated Learning Security (Flower + PyTorch on CIFAR-10):**
• Benchmarked **11 full 35-round simulations** across `FedAvg`, `FedAvgM`, `FedMedian`, and `FedTrimmedAvg` under 0%, 20%, and 40% label-flipping attackers.
• Demonstrated why server momentum (`FedAvgM`) worsens poisoning by accumulating malicious gradients in the velocity vector.
• Showed the exact trade-off between **Trimmed Mean (`FedTrimmedAvg`)**—which wins at low attacker ratios (**72.32%** at 20% poisoning)—and **Coordinate-wise Median (`FedMedian`)**, which dominates at high attacker ratios (**59.67%** vs. `FedAvg`'s **44.76%** at 40% poisoning) thanks to its 50% breakdown point.

Check out the full benchmark, visualizations, and code on GitHub:
👉 https://github.com/boostx1/Security-in-AI-systems

#AISecurity #FederatedLearning #MachineLearning #DeepLearning #PyTorch #AdversarialML #AIEngineering

---

## Recommended Images to Attach to Your LinkedIn Post

Upload these 3 images from the `images/` folder as a multi-image carousel on LinkedIn:
1. `images/part1_pgd_attack_visualization.png` *(Shows the Koala -> Tractor PGD attack and amplified noise)*
2. `images/part2_4_trimmed_mean_defense.png` *(Shows the head-to-head comparison of FedTrimmedAvg vs. FedMedian vs. FedAvg)*
3. `images/part2_2_fedmedian_defense.png` *(Shows the 35-round Accuracy & Cohen's Kappa recovery under FedMedian)*
