# test_fine_tuned.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "fine_tuned_model"

# Load tokenizer and model
print("Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.float16)

# Define the prompt
prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
What is the employee's role in the company?

### Input:
Employee: Emily Davis

### Response:"""

# Tokenize the input and move to GPU
print("Generating response...")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# Generate the output
outputs = model.generate(**inputs, max_new_tokens=128)
# Decode and print the output
print("Generated response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
