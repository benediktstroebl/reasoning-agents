import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from trl import setup_chat_format
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ModelConfig:
    base_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    finetuned_path: str = "models/llama-3.2-3b-ft-max-seq-length/final_model"
    use_peft: bool = True
    torch_dtype: torch.dtype = torch.float16
    max_new_tokens: int = 8000
    base_system_prompt: str = "You are a helpful AI assistant."
    finetuned_system_prompt: str = "# Retail agent policy\n\nAs a retail agent, you can help users cancel or modify pending orders..."

def load_models(config: ModelConfig):        
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=config.torch_dtype,
        device_map="auto",
    )
    
    
    # Load finetuned model
    if config.use_peft:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            config.finetuned_path,
            low_cpu_mem_usage=True,
            return_dict=True, 
            torch_dtype=config.torch_dtype,
            device_map="auto",
        )
        finetuned_model = PeftModel.from_pretrained(finetuned_model, config.finetuned_path)
        finetuned_model = finetuned_model.merge_and_unload()
    else:
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            config.finetuned_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=config.torch_dtype, 
            device_map="auto"
        )
        
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = base_model.config.eos_token_id
    
    return base_model, finetuned_model, tokenizer

def generate(model, tokenizer, prompt, config: ModelConfig, is_base_model: bool = False):
    messages = [
        {"role": "system", "content": config.base_system_prompt if is_base_model else config.finetuned_system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        num_return_sequences=1
    )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("assistant")[-1].strip()

def main():
    config = ModelConfig()
    
    base_model, finetuned_model, tokenizer = load_models(config)
    
    while True:
        prompt = input("\nEnter your prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
            
        print("\nBase model response:")
        print(generate(base_model, tokenizer, prompt, config, is_base_model=True))
        
        print("\nFinetuned model response:") 
        print(generate(finetuned_model, tokenizer, prompt, config))

if __name__ == "__main__":
    main() 