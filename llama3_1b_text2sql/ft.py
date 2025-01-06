import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Model ID
modelID = "NousResearch/Llama-3.2-1B"

# Load the dataset
dataset = load_dataset("lamini/spider_text_to_sql", split="train")

# Inspect the dataset structure
print("Columns:", dataset.column_names)  # Output: ['input', 'output']

# Ensure the dataset has enough samples
if len(dataset) == 0:
    raise ValueError("The dataset is empty or improperly loaded.")

# Filter out rows with empty "input" or "output"
dataset = dataset.filter(lambda example: len(example["input"]) > 0 and len(example["output"]) > 0)

# Preprocess the dataset to combine input and output
def preprocess(example):
    return {
        "text": f"<s> {example['input']}  {example['output']} </s>"
    }

# Map preprocessing function over the dataset
dataset = dataset.map(preprocess)
print("First sample:", dataset[0])  # Check structure of the dataset

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
    output_dir="./results_sql_1b",
    num_train_epochs = 10
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=sft_config.max_seq_length,
    dataset_text_field="text",
    packing=sft_config.packing,
    args=sft_config,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./results_sql_1b")

# Inference pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Example input for SQL generation
example_input = (
    "[INST] Here is a database schema: flight : flno [ INT ] primary_key origin [ TEXT ] destination [ TEXT ] "
    "distance [ INT ] departure_date [ TEXT ] arrival_date [ TEXT ] price [ INT ] aid [ INT ] "
    "flight.aid = aircraft.aid aircraft : aid [ INT ] primary_key name [ TEXT ] distance [ INT ] "
    "employee : eid [ INT ] primary_key name [ TEXT ] salary [ INT ] certificate : eid [ INT ] primary_key "
    "certificate.eid = employee.eid aid [ INT ] certificate.aid = aircraft.aid Please write me a SQL statement "
    "that answers the following question: What is the salary and name of the employee who has the most number "
    "of certificates on aircrafts with distance more than 5000? [/INST]"
)

# Generate text
sequences = generation_pipeline(
    example_input,
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1
)

# Print results
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
