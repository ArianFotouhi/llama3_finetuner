import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Model ID
modelID = "NousResearch/Llama-3.2-1B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelID)

# Load untrained model
untrained_model = AutoModelForCausalLM.from_pretrained(modelID)

# Load fine-tuned model directly
fine_tuned_model_path = "./results_sql_1b"
fine_tuned_model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)

# Define inference pipelines
untrained_pipeline = pipeline("text-generation", model=untrained_model, tokenizer=tokenizer, device_map="auto")
fine_tuned_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer, device_map="auto")

# Input prompt for comparison
example_input = (
    "[INST] Here is a database schema: flight : flno [ INT ] primary_key origin [ TEXT ] destination [ TEXT ] "
    "distance [ INT ] departure_date [ TEXT ] arrival_date [ TEXT ] price [ INT ] aid [ INT ] "
    "flight.aid = aircraft.aid aircraft : aid [ INT ] primary_key name [ TEXT ] distance [ INT ] "
    "employee : eid [ INT ] primary_key name [ TEXT ] salary [ INT ] certificate : eid [ INT ] primary_key "
    "certificate.eid = employee.eid aid [ INT ] certificate.aid = aircraft.aid Please write me a SQL statement "
    "that answers the following question: What is the name of the aircraft with the highest average flight price? [/INST]"
)

# Generate output from untrained model
print("Generating output from untrained model...")
untrained_output = untrained_pipeline(
    example_input, max_length=400, do_sample=True, top_k=10, num_return_sequences=1
)[0]["generated_text"]

# Generate output from fine-tuned model
print("Generating output from fine-tuned model...")
fine_tuned_output = fine_tuned_pipeline(
    example_input, max_length=400, do_sample=True, top_k=10, num_return_sequences=1
)[0]["generated_text"]

# Display results
print("\n=== Comparison of Outputs ===")
print("\nInput Prompt:")
print(example_input)
print("\nUntrained Model Output:")
print(untrained_output)
print("\nFine-Tuned Model Output:")
print(fine_tuned_output)
