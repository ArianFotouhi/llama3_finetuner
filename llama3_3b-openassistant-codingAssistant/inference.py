import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Define paths and configurations
model_path = "./results"  # Path to the fine-tuned model

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Define the inference pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"  # Use available GPU/CPU resources
)

# Input prompt for generation
input_prompt = (
    "Can you create a function in Javascript that returns a string of the current date in this format? 6 Februray 2023\nSam:"

)

# Generate text
generated_sequences = text_generator(
    input_prompt,
    max_length=200,  # Limit the length of the generated text
    do_sample=True,  # Enable sampling for varied responses
    top_k=10,  # Consider top-k tokens for sampling
    num_return_sequences=1  # Generate one response
)

# Print the generated text
for idx, seq in enumerate(generated_sequences):
    print(f"Generated Text {idx + 1}: {seq['generated_text']}")
