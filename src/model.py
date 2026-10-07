import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(adapter_id=None, model_id="Qwen/Qwen2.5-1.5B-Instruct"):

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )

    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)

    return model, tokenizer
