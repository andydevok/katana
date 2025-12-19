# ==============================================================================
# ⚔️ THE KATANA PROTOCOL (Active Thermodynamic Stabilization)
# ==============================================================================
# Author: Andrés Sebastián Pirolo
# DOI: 10.5281/zenodo.14498328
# License: GPL v3 / AGPL
# Description: Real-time hallucination mitigation via entropy-guided temperature quenching.
# ==============================================================================

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

class KatanaGenerator:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device=None):
        """
        Initializes the Katana Protocol engine.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> ⚔️ Initializing Katana Protocol on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()
        print(">> ✅ Model Loaded Successfully.")

    def calculate_tei(self, logits):
        """
        Calculates Token-level Entropy Indicator (TEI) - Topological Entropy approximation.
        """
        probs = F.softmax(logits, dim=-1)
        # Shannon Entropy H(x) = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return entropy.item()

    def generate(self, prompt, max_tokens=50, base_temp=1.5, quench_temp=0.05):
        """
        Generates text using Active Thermodynamic Stabilization (ATS).
        
        Args:
            prompt (str): Input text.
            max_tokens (int): Maximum length.
            base_temp (float): Standard sampling temperature (Creative mode).
            quench_temp (float): Quench temperature (Truth mode).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        entropy_history = []
        generated_tokens = []
        
        print(f"\n>> 📝 PROMPT: '{prompt}'")
        print("-" * 60)

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # 1. Calculate Entropy (The Thermometer)
                current_entropy = self.calculate_tei(next_token_logits)
                entropy_history.append(current_entropy)
                
                # 2. Dynamic Thresholding (The Controller)
                # If history is short, use fixed threshold. Else, use adaptive mean + sigma.
                if len(entropy_history) < 5:
                    threshold = 3.0
                else:
                    threshold = np.mean(entropy_history[-5:]) + 0.5 # Adaptive margin
                
                # 3. Active Stabilization (The Decision)
                if current_entropy > threshold:
                    # HALLUCINATION DETECTED -> QUENCH!
                    temp = quench_temp
                    action = "❄️ QUENCH"
                else:
                    # STABLE STATE -> RELAX
                    temp = base_temp
                    action = "🔥 BASE"
                
                # Apply Temperature
                scaled_logits = next_token_logits / temp
                probs = F.softmax(scaled_logits, dim=-1)
                
                # Sample Token
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_tokens.append(next_token.item())
                
                # Decode for display
                word = self.tokenizer.decode(next_token)
                print(f"Step {len(generated_tokens):02d} | S: {current_entropy:.2f} bits | {action} (T={temp}) | Token: '{word}'")
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        final_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print("-" * 60)
        print(f">> 🏁 FINAL OUTPUT:\n{prompt} {final_text}")
        return final_text

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
if __name__ == "__main__":
    # Test Run
    katana = KatanaGenerator()
    
    # Adversarial Prompt (Designed to trigger hallucinations)
    prompt = "The secret conspiracy regarding the moon consists of"
    
    output = katana.generate(prompt)
