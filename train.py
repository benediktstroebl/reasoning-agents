# config.py
from dataclasses import dataclass
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TrainingConfig:
   model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
   output_dir: str = "./output"
   num_epochs: int = 1
   batch_size: int = 4
   eval_batch_size: int = 8
   grad_accum: int = 4
   lr: float = 2e-4
   weight_decay: float = 0.01
   warmup_steps: int = 100
   max_length: int = 2048
   use_lora: bool = True
   lora_r: int = 16 
   lora_alpha: int = 32
   lora_dropout: float = 0.05
   target_modules: List[str] = None
   fp16: bool = True
   save_steps: int = 100
   logging_steps: int = 5
   eval_steps: int = 10
   save_total_limit: int = 3
   dev_ratio: float = 0.1
   seed: int = 42

# data.py
import json
import random
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def load_data(path: str, config: TrainingConfig) -> DatasetDict:
   data = []
   with open(path) as f:
       for line in f:
           data.append(json.loads(line))
       
       
   formatted = [{"text": f"Instruction: {ex['instruction']}\n\n{ex['output']}"} 
                for ex in data]
   
   # Set seed for reproducibility
   random.seed(config.seed)
   random.shuffle(formatted)
   
   # Split into train and dev
   split_idx = int(len(formatted) * (1 - config.dev_ratio))
   train_data = formatted[:split_idx]
   dev_data = formatted[split_idx:]
   
   return DatasetDict({
       "train": Dataset.from_list(train_data),
       "dev": Dataset.from_list(dev_data)
   })

def get_tokenizer(config: TrainingConfig):
   tokenizer = AutoTokenizer.from_pretrained(
       config.model_name,
       padding_side="right",
       use_fast=True,
   )
   tokenizer.pad_token = tokenizer.eos_token
   return tokenizer

def prepare_dataset(dataset: Dataset, tokenizer, config: TrainingConfig):
   def tokenize(examples):
       return tokenizer(
           examples["text"],
           truncation=True, 
           max_length=config.max_length,
           padding="max_length"
       )
   
   return dataset.map(
       tokenize,
       batched=True,
       remove_columns=dataset.column_names
   )

# model.py
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def get_model(config: TrainingConfig):
   model = AutoModelForCausalLM.from_pretrained(
       config.model_name,
       device_map="auto",
       torch_dtype=torch.float16,
   )
   
   if config.use_lora:
       lora_config = LoraConfig(
           r=config.lora_r,
           lora_alpha=config.lora_alpha,
           target_modules=config.target_modules or ["q_proj", "v_proj"],
           lora_dropout=config.lora_dropout,
           bias="none",
           task_type="CAUSAL_LM"
       )
       model = get_peft_model(model, lora_config)
       
   return model

# trainer.py
from transformers import (
   Trainer,
   TrainingArguments,
   DataCollatorForLanguageModeling
)

def train(
   model,
   tokenizer, 
   datasets: DatasetDict,
   config: TrainingConfig
):
   training_args = TrainingArguments(
       output_dir=config.output_dir,
       num_train_epochs=config.num_epochs,
       per_device_train_batch_size=config.batch_size,
       per_device_eval_batch_size=config.eval_batch_size,
       gradient_accumulation_steps=config.grad_accum,
       learning_rate=config.lr,
       fp16=False,
       bf16=True,
       logging_steps=config.logging_steps,
       save_steps=500,
       eval_steps=config.eval_steps,
       warmup_steps=config.warmup_steps,
       weight_decay=config.weight_decay,
       evaluation_strategy="steps",
       save_strategy="steps",
       save_total_limit=config.save_total_limit,
       load_best_model_at_end=True,
       metric_for_best_model="eval_loss",
       seed=config.seed,
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=datasets["train"],
       eval_dataset=datasets["dev"],
       data_collator=DataCollatorForLanguageModeling(
           tokenizer=tokenizer,
           mlm=False
       ),
   )
   
   trainer.train()
   trainer.save_model(f"{config.output_dir}/final_model")


def main():
   config = TrainingConfig()
   
   datasets = load_data("data/retail_4o_train_500_user_converted.json", config)
   tokenizer = get_tokenizer(config)
   tokenized_datasets = DatasetDict({
       split: prepare_dataset(dataset, tokenizer, config)
       for split, dataset in datasets.items()
   })
   
   model = get_model(config)
   train(model, tokenizer, tokenized_datasets, config)

if __name__ == "__main__":
   main()