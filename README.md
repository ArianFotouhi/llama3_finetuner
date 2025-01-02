# Coding Chatbot with Fine-Tuned Language Model

## Overview
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

## Llama3 3B Example

```
Prompt: Can you create a function in Javascript that returns a string of the current date in this format? 6 Februray 2023
```

#### Before Fine-Tune

function getCurrentDate() {\n    const today = new Date();\n    const options = { weekday: '"}]

#### After Fine-Tune
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
