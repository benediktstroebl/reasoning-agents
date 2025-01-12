from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)


from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format, SFTConfig
from dataclasses import dataclass
from typing import Optional, List

from huggingface_hub import login

import json
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

wb_token = os.getenv("WANDB_API_KEY")

wandb.login(key=wb_token)
run = wandb.init(
    project='Fine-tune Llama 3.2', 
    job_type="training", 
    anonymous="allow",
    name="runname"
)

@dataclass 
class TrainingConfig:
    run_name: str = "runname"
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    new_model: str = "llama-3.2-3b-ft-max-seq-length"
    num_train_samples: int = 390
    use_lora: bool = True
    
    # Training params
    num_epochs: int = 1
    batch_size: int = 2
    eval_batch_size: int = 1
    grad_accum: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 4096
    
    # LoRA params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Other
    seed: int = 65
    eval_steps: float = 0.2
    logging_steps: int = 1
    
    dev_split: float = 0.1
    
    def __post_init__(self):
        if torch.cuda.get_device_capability()[0] >= 8:
            self.torch_dtype = torch.bfloat16
            self.attn_implementation = "flash_attention_2"
        else:
            self.torch_dtype = torch.float16
            self.attn_implementation = "eager"
            
            
import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



def load_data(path: str, config: TrainingConfig, tokenizer):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
            
    dataset = Dataset.from_list(data)
    dataset = dataset.shuffle(seed=config.seed).select(range(config.num_train_samples))
        
    def format_chat_template(row):
        row_json = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["instruction"]}, 
            {"role": "assistant", "content": row["output"]}
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=4,
    )
    
    # Get max sequence length from data
    max_length = max(len(tokenizer.encode(example["text"])) for example in dataset)
    config.max_seq_length = max_length
    if max_length > 8000:
        config.max_seq_length = 4096
    print(f"Max sequence length set to: {config.max_seq_length}")
    #     raise ValueError(f"Max sequence length {max_length} is too large. Please reduce the number of train samples or the length of the input data.")
    
    # Split into train and validation
    dataset = dataset.train_test_split(test_size=config.dev_split)
    
    # print one example from train
    print(dataset["train"][0]["text"])
    
    return dataset

def main():
    config = TrainingConfig()
        
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=config.torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config if config.use_lora else None,
        device_map="auto",
        attn_implementation=config.attn_implementation,
        torch_dtype=config.torch_dtype,
    )
    
    modules = find_all_linear_names(model)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # Dataset loading
    dataset = load_data("data/retail_4o_train_500_cot.json", config, tokenizer)

    # Only apply LoRA if use_lora is True
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=modules
        )
        model = get_peft_model(model, peft_config)
    
    training_arguments = SFTConfig(
        output_dir=f"models/{config.new_model}",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.grad_accum,
        optim="paged_adamw_8bit",
        num_train_epochs=config.num_epochs,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        warmup_steps=config.warmup_steps,
        logging_strategy="steps",
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb",
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
        run_name=config.run_name
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config if config.use_lora else None,
        processing_class=tokenizer,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_model(f"models/{config.new_model}/final_model")
    wandb.finish()

if __name__ == "__main__":
    main()
