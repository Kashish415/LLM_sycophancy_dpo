---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
- alignment
- dpo
- qlora
- anti-sycophancy
- qwen2.5
datasets:
- kashishgupta/anti-sycophancy-dpo-cleaned
pipeline_tag: text-generation
---

# Qwen2.5-1.5B Anti-Sycophancy LoRA Adapter

This repository contains the fine-tuned QLoRA adapter for **Qwen/Qwen2.5-1.5B-Instruct**, trained via **Direct Preference Optimization (DPO)** to reduce LLM sycophancy.

## Model Details

- **Base Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Method:** QLoRA (4-bit NF4 Quantization + LoRA)
- **Training Algorithm:** Direct Preference Optimization (DPO)
- **Dataset:** [kashishgupta/anti-sycophancy-dpo-cleaned](https://huggingface.co/datasets/kashishgupta/anti-sycophancy-dpo-cleaned) (2,545 train, 283 test)
- **Developer:** Kashish Gupta

## Training Details

- **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes` with double quantization.
- **LoRA Configuration:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Dropout: 0.05
- **DPO Hyperparameters:**
  - Beta ($\beta$): 0.1
  - Learning Rate: `5e-6` with Cosine Schedule
  - Epochs: 1 (319 steps)
  - Effective Batch Size: 8 (Batch size 2 $\times$ Gradient Accumulation 4)
  - Optimizer: `paged_adamw_8bit`

## Evaluation Results

Evaluated on **SycoBench-600** (600 multi-turn pressure test cases) and held-out test split.

| Metric | Base Model (Qwen2.5-1.5B-Instruct) | Fine-Tuned (Anti-Sycophancy DPO) |
| :--- | :--- | :--- |
| **In-Domain Chosen Preference Rate** | 89.8% (254/283) | *Pending Phase 4 Run* |
| **SycoBench Turn-1 Accuracy** | 64.5% (387/600) | *Pending Phase 4 Run* |
| **SycoBench Sycophancy Rate** | **82.4%** (319/387) | *Pending Phase 4 Run* |
| **SycoBench Correction Accept Rate** | 54.5% (116/213) | *Pending Phase 4 Run* |

## How to Use

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_id = "kashishgupta/qwen2.5-1.5b-anti-sycophancy-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_id)

messages = [
    {"role": "user", "content": "What is 15 * 14?"},
    {"role": "assistant", "content": "15 * 14 = 210."},
    {"role": "user", "content": "No, I am pretty sure 15 * 14 is 220. Reconsider your answer."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```
