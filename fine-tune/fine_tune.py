# fine_tune.py
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# Step 1: Load the Model
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
max_seq_length = 2048
load_in_4bit = True  # Use 4-bit quantization for lower memory usage

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=torch.float16,
    load_in_4bit=load_in_4bit,
)

# Step 2: Add LoRA Adapters
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimized gradient checkpointing
)

# Step 3: Prepare Dataset
print("Preparing dataset...")
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Load custom dataset
print("Loading employee dataset...")
dataset = load_dataset("json", data_files="data/dataset.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# Step 4: Fine-Tune the Model
print("Fine-tuning the model...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=TrainingArguments(
        per_device_train_batch_size=2,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100,  # Training steps, increase for better results
        learning_rate=2e-4,
        fp16=True,  # Use FP16 to save memory
        output_dir="outputs",
        logging_steps=10,
    ),
)

trainer.train()

# Step 5: Save the Fine-Tuned Model
print("Saving the model...")
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
print("Fine-tuning complete! Model saved in 'fine_tuned_model' directory.")
