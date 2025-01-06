# Fine-Tuned Language Model

## Llama3 3B (Coding Chatbot)

This project creates a **coding chatbot** using a pre-trained language model (`Hermes-3-Llama`). The chatbot is fine-tuned on the OpenAssistant dataset to improve its ability to answer coding-related queries and generate natural, context-aware responses.

## Features
- **Quantization for Efficiency**: Uses 4-bit quantization to reduce memory usage while maintaining performance.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Implements Low-Rank Adaptation (LoRA) to fine-tune specific layers without retraining the entire model.
- **Custom Tokenizer**: Adds special tokens like `<PAD>` for handling sequence padding during training.
- **Inference Pipeline**: Generates conversational text outputs for user queries.

## Workflow
1. **Load Dataset**: The model is fine-tuned on the `openassistant-guanaco` dataset for conversational understanding.
2. **Model Preparation**:
   - A pre-trained causal language model is loaded (`Hermes-3-Llama`).
   - Quantization reduces computational and memory costs.
3. **Fine-Tuning**:
   - LoRA focuses training on key attention layers (`q_proj` and `v_proj`).
   - The SFTTrainer handles supervised fine-tuning with configuration for sequence length, packing, and data handling.
4. **Inference**:
   - A `pipeline` for text generation is set up to handle user inputs and generate responses.

## Requirements
- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets library
- TRL (Transformers Reinforcement Learning)

Install dependencies:
```bash
pip install torch transformers datasets trl peft
```



```
Prompt: Can you create a function in Javascript that returns a string of the current date in this format? 6 Februray 2023
```

#### Before Fine-Tune:

function getCurrentDate() {\n    const today = new Date();\n    const options = { weekday: '"}]


-------------------------------------------------
#### After Fine-Tune:
I can create a function in JavaScript that returns a string of the current date in this format, 6 February 2023. Here's an example:

```javascript
function getCurrentDate() {
  const today = new Date();
  const day = today.getDate();
  const month = today.getMonth() + 1; // Month is zero-indexed
  const year = today.getFullYear();

  const formattedDate = `${day} ${month} ${year}`;
  return formattedDate;
}
```

You can call the function like this:

```javascript
console.log(getCurrentDate());
```
-------------------------------------------------


## Llama3 1B (SQL agent)

This repository contains code and models for fine-tuning a language model on the [Lamini/Spider Text-to-SQL Dataset](https://huggingface.co/datasets/lamini/spider_text_to_sql) to generate SQL queries based on natural language inputs. The project demonstrates fine-tuning, inference, and comparison of outputs between a pre-trained base model and the fine-tuned model.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Fine-Tuning](#fine-tuning)
- [Inference Pipeline](#inference-pipeline)
- [Comparison of Outputs](#comparison-of-outputs)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

The goal of this project is to fine-tune a pre-trained causal language model, [NousResearch/Llama-3.2-1B](https://huggingface.co/NousResearch/Llama-3.2-1B), on a structured SQL generation dataset. The model learns to generate accurate SQL queries based on given natural language questions and database schema.

---

## Dataset

The [Lamini/Spider Text-to-SQL Dataset](https://huggingface.co/datasets/lamini/spider_text_to_sql) is used for training. It includes:
- Database schemas
- Corresponding natural language questions
- Ground-truth SQL queries

---

## Fine-Tuning

The fine-tuning process adjusts the model to better understand SQL structures and generate precise queries. Training scripts and configurations are included in the repository.

Fine-tuned model directory: `./results_sql_1b`

---

## Inference Pipeline

The repository includes an example pipeline for generating SQL queries using both the base and fine-tuned models.



# Load tokenizer and models
modelID = "NousResearch/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(modelID)
untrained_model = AutoModelForCausalLM.from_pretrained(modelID)
fine_tuned_model = AutoModelForCausalLM.from_pretrained("./results_sql_1b")

# Define pipelines
untrained_pipeline = pipeline("text-generation", model=untrained_model, tokenizer=tokenizer, device_map="auto")
fine_tuned_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer, device_map="auto")

# Example input prompt
example_input = "[INST] ... SQL question here ... [/INST]"

# Generate outputs
untrained_output = untrained_pipeline(example_input, max_length=400, do_sample=True)[0]["generated_text"]
fine_tuned_output = fine_tuned_pipeline(example_input, max_length=400, do_sample=True)[0]["generated_text"]

# Compare results
print("Untrained Model Output:", untrained_output)
print("Fine-Tuned Model Output:", fine_tuned_output)

Comparison of Outputs

Input Prompt:
```bash
[INST] Here is a database schema: flight : flno [ INT ] primary_key origin [ TEXT ] destination [ TEXT ] distance [ INT ] departure_date [ TEXT ] arrival_date [ TEXT ] price [ INT ] aid [ INT ] flight.aid = aircraft.aid aircraft : aid [ INT ] primary_key name [ TEXT ] distance [ INT ] employee : eid [ INT ] primary_key name [ TEXT ] salary [ INT ] certificate : eid [ INT ] primary_key certificate.eid = employee.eid aid [ INT ] certificate.aid = aircraft.aid Please write me a SQL statement that answers the following question: What is the name of the aircraft with the highest average flight price? [/INST]
```

Outputs:

Untrained Model:

```bash
[INST] Here is a database schema: flight : flno [ INT ] primary_key origin [ TEXT ] destination [ TEXT ] distance [ INT ] departure_date [ TEXT ] arrival_date [ TEXT ] price [ INT ] aid [ INT ] flight.aid = aircraft.aid aircraft : aid [ INT ] primary_key name [ TEXT ] distance [ INT ] employee : eid [ INT ] primary_key name [ TEXT ] salary [ INT ] certificate : eid [ INT ] primary_key certificate.eid = employee.eid aid [ INT ] certificate.aid = aircraft.aid Please write me a SQL statement that answers the following question: What is the name of the aircraft with the highest average flight price? [/INST]
```
The untrained model only repeats the input prompt


Fine-Tuned Model:
```bash
SELECT T1.name
FROM aircraft AS T1
JOIN flight AS T2 ON T1.aid = T2.aid
GROUP BY T1.aid
ORDER BY avg(T2.price) DESC
LIMIT 1;
```



