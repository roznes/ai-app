###### README.md
### Requirements
# Python (WSL-Ubuntu)
- **Version**: `Python 3.12.7`

========================================

### **Initial Setup for WSL-Ubuntu**
[Terminal]
```sh
sudo apt update
# GPU Setup
sudo apt install nvidia-cuda-toolkit -y
nvcc --version
nvidia-smi

# Python Setup
python3 --version
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
sudo apt install python-is-python3 -y
python --version
pip --version
```

---

### **Set Up Python Virtual Environment**
[Terminal]
```sh
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install PyTorch
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### **Install Required Python Packages**
[Terminal]
```sh
pip install transformers datasets accelerate bitsandbytes trl peft unsloth
```

**Explanation of Packages**:
1. **`transformers`**: For working with Hugging Face's transformer models.
2. **`datasets`**: To handle datasets for fine-tuning, including loading and preprocessing.
3. **`accelerate`**: Provides tools for distributed training and multi-GPU support.
4. **`bitsandbytes`**: Enables 4-bit and 8-bit quantization for memory-efficient training.
5. **`trl`**: Includes fine-tuning tools like `SFTTrainer` for instruction tuning and preference modeling.
6. **`peft`**: For parameter-efficient fine-tuning, including LoRA.
7. **`unsloth`**: The main library for 2x faster training and lower memory usage with LLMs.

---

### **Validate Installation**
[Terminal]
```sh
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

#### run venv (wsl-ubuntu)
[Terminal]
```sh
python3 -m venv venv
source venv/bin/activate

python fine_tune.py
python test_fine_tuned.py
```