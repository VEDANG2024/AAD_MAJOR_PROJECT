import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# 1. Configuration
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" # The Student SLM
NEW_MODEL_NAME = "Phi-3-UGAD-Lite-Student"

# 2. QLoRA Config (The "Lite" part)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 3. Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4. Prepare for Training
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16, 
    lora_alpha=16, 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

# 5. Load Data (GSM8K)
dataset = load_dataset("gsm8k", "main", split="train[:500]") # Small subset for demo
def format_prompts(examples):
    # Formats data as "Question -> Answer"
    texts = []
    for q, a in zip(examples['question'], examples['answer']):
        text = f"<|user|>\n{q} <|end|>\n<|assistant|>\n{a} <|end|>"
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(format_prompts, batched=True)

# 6. Train!
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=50, # Short run for demo purposes
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
)

print("Starting UGAD-Lite Training...")
trainer.train()
print("Training Complete! Model saved.")
trainer.model.save_pretrained(NEW_MODEL_NAME)
