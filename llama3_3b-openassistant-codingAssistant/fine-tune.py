import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Model ID
modelID = "NousResearch/Hermes-3-Llama-3.2-3B"

# Load dataset
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# Quantization configuration
quantizationConfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(modelID, quantization_config=quantizationConfig)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelID)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

# PEFT LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training configuration
sft_config = SFTConfig(
    max_seq_length=512,
    packing=True,
    output_dir="./results"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=sft_config.max_seq_length,
    dataset_text_field="text",
    packing=sft_config.packing
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./results")

# Inference pipeline
from transformers import pipeline
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Generate text
sequences = pipeline(
   "Can you create a function in Javascript that returns a string of the current date in this format? 6 Februray 2023\nSam:",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1
)

# Print results
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
