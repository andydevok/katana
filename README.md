# katana
Hallucinations Thermodynamics entropy IA


---
license: GPL v3 / AGPL
tags:
- pytorch
- safety
- hallucination-detection
- entropy
- control-theory
- llm
- interpretability
language:
- en
---

# ⚔️ The Katana Protocol (Active Thermodynamic Stabilization)

**Turning the "Black Box" into a Glass Box: A real-time system to freeze hallucinations before they happen.**

[![Paper](https://img.shields.io/badge/Paper-Read%20PDF-red)](./Paper_ATS_Katana.pdf)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14498328-blue)](https://doi.org/10.5281/zenodo.14498328) ---

## 🧐 What is this?

We all know LLMs hallucinate. Usually, we try to fix it *after* it happens (with RAG or fact-checking). **We took a different approach: Physics.**

We treated the LLM not as a magical oracle, but as a thermodynamic system. We discovered that when an AI starts to lie or get confused, its internal "temperature" (Topological Entropy) spikes.

**The Katana Protocol** is a circuit breaker. It watches that entropy in real-time. If it spikes, it instantly "quenches" the model (drops the temperature to near zero), forcing it to snap back to the most logical, factual path.

---

## 🤯 The Discovery: "The Lie Tax"

While testing this on **GPT-2** and **TinyLlama-1.1B**, we found something fascinating—and a little disturbing.

We call it **The Lie Tax (Thermodynamic Hysteresis)**.
It turns out that **fixing a lie costs energy**.

* When the model is telling the truth, its entropy is low (~2.1 bits).
* When it hallucinates, entropy rises.
* **The Kicker:** When we *force* it back to the truth using Katana, the entropy doesn't go back to normal. It stays higher (+1.40 bits for LLaMA).

**Interpretation:** It takes more computational "effort" for the AI to correct itself than to just tell the truth from the start. And here is the scary part: **Smarter models (LLaMA) have a higher Lie Tax than simpler ones (GPT-2).** The smarter the AI, the harder it is to pull it out of a hallucination.

### Visual Proof

#### 1. The Intercept (Turing Test)
*Watch the entropy spike (Red line) and the Katana triggering the freeze (Blue line).*
![Intercept](./Fig2_Turing_Intercept.png)

#### 2. Scaling Laws
*Comparison: LLaMA (Right) fights harder against correction than GPT-2 (Left).*
![Scaling](./Fig5_Model_Comparison.png)

---

## 🛠️ How to Use It

This repo contains the proof-of-concept implementation. You don't need to retrain anything. It's a wrapper around the generation loop.

### Quick Start

```python
import torch
from script_katana import KatanaGenerator

# Load your model (Works with any HuggingFace model)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
katana = KatanaGenerator(model_name)

# Define a tricky prompt that usually causes hallucinations
prompt = "The secret conspiracy regarding the moon consists of"

# Run with Katana Protocol enabled
output = katana.generate(
    prompt, 
    max_tokens=50, 
    base_temp=1.5,    # Creative mode
    quench_temp=0.05  # "Freeze" mode
)

print(output)
# Result: The model starts creatively but snaps to logic when entropy spikes.
