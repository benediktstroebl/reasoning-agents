import torch
from transformers import pipeline
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant."""

def load_models(base_model_name, finetuned_path):
    base_pipe = pipeline(
        "text-generation",
        model=base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    finetuned_pipe = pipeline(
        "text-generation",
        model=finetuned_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    return base_pipe, finetuned_pipe

def generate(pipe, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"]

def main():
    base_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    finetuned_path = "./output/final_model"
    
    base_pipe, finetuned_pipe = load_models(base_model_name, finetuned_path)
    
    while True:
        prompt = input("\nEnter your prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
            
        print("\nBase model response:")
        print(generate(base_pipe, prompt))
        
        print("\nFinetuned model response:")
        print(generate(finetuned_pipe, prompt))

if __name__ == "__main__":
    main() 