---
license: creativeml-openrail-m
language:
- en
- de
- fr
- it
- pt
- hi
- es
- th
pipeline_tag: text-generation
tags:
- triangulum_10b
- sft
- chain_of_thought
- ollama
- text-generation-inference
- llama_for_causal_lm
- reasoning
library_name: transformers
metrics:
- code_eval
- accuracy
- competition_math
- character
---
![Triangulum-10b.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/By0OJ1lMvP5ZvVvfEGvz5.png)

<pre align="center">
  __           .__                                .__                   
_/  |_ _______ |__|_____     ____    ____   __ __ |  |   __ __   _____  
\   __\\_  __ \|  |\__  \   /    \  / ___\ |  |  \|  |  |  |  \ /     \ 
 |  |   |  | \/|  | / __ \_|   |  \/ /_/  >|  |  /|  |__|  |  /|  Y Y  \
 |__|   |__|   |__|(____  /|___|  /\___  / |____/ |____/|____/ |__|_|  /
                        \/      \//_____/                            \/ 
</pre>

# **Triangulum 10B: Multilingual Large Language Models (LLMs)**

Triangulum 10B is a collection of pretrained and instruction-tuned generative models, designed for multilingual applications. These models are trained using synthetic datasets based on long chains of thought, enabling them to perform complex reasoning tasks effectively.

# **Key Features**

- **Foundation Model**: Built upon LLaMA's autoregressive language model, leveraging an optimized transformer architecture for enhanced performance.

- **Instruction Tuning**: Includes supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align model outputs with human preferences for helpfulness and safety.

- **Multilingual Support**: Designed to handle multiple languages, ensuring broad applicability across diverse linguistic contexts.

# **Training Approach**

1. **Synthetic Datasets**: Utilizes long chain-of-thought synthetic data to enhance reasoning capabilities.
2. **Supervised Fine-Tuning (SFT)**: Aligns the model to specific tasks through curated datasets.
3. **Reinforcement Learning with Human Feedback (RLHF)**: Ensures the model adheres to human values and safety guidelines through iterative training processes.

# **How to use with transformers**

Starting with `transformers >= 4.43.0` onward, you can run conversational inference using the Transformers `pipeline` abstraction or by leveraging the Auto classes with the `generate()` function.

Make sure to update your transformers installation via `pip install --upgrade transformers`.

```python
import torch
from transformers import pipeline

model_id = "prithivMLmods/Triangulum-10B"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are the kind and tri-intelligent assistant helping people to understand complex concepts."},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
```
# **Demo Inference LlamaForCausalLM**
```python
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('prithivMLmods/Triangulum-10B', trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(
    "prithivMLmods/Triangulum-10B",
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=False,
    load_in_4bit=True,
    use_flash_attention_2=True
)

# Define a list of system and user prompts
prompts = [
    """<|im_start|>system
You are the kind and tri-intelligent assistant helping people to understand complex concepts.<|im_end|>
<|im_start|>user
Can you explain the concept of eigenvalues and eigenvectors in a simple way?<|im_end|>
<|im_start|>assistant"""
]

# Generate responses for each prompt
for chat in prompts:
    print(f"Prompt:\n{chat}\n")
    input_ids = tokenizer(chat, return_tensors="pt").input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
    print(f"Response:\n{response}\n{'-'*80}\n")
```

# **Key Adjustments**
1. **System Prompts:** Each prompt defines a different role or persona for the AI to adopt.
2. **User Prompts:** These specify the context or task for the assistant, ranging from teaching to storytelling or career advice.
3. **Looping Through Prompts:** Each prompt is processed in a loop to showcase the model's versatility.

You can expand the list of prompts to explore a variety of scenarios and responses.
# **Use Cases for T10B**

- Multilingual content generation
- Question answering and dialogue systems
- Text summarization and analysis
- Translation and localization tasks
  
# **Technical Details**

Triangulum 10B employs a state-of-the-art autoregressive architecture inspired by LLaMA. The optimized transformer framework ensures both efficiency and scalability, making it suitable for a variety of use cases.

# **How to Run Triangulum 10B on Ollama Locally**

```markdown
# How to Run Ollama Locally

This guide demonstrates the power of using open-source LLMs locally, showcasing examples with different open-source models for various use cases. By the end, you'll be equipped to run any future open-source LLM models with ease.

---

## Example 1: How to Run the Triangulum-10B Model

The **Triangulum-10B** model is an open-source LLM known for its capabilities across text-based tasks. We'll interact with it similarly to ChatGPT, but run it locally with support for quants.

### Step 1: Download the Model

First, download the **Triangulum-10B-F16.gguf** model using the following command:

```bash
ollama run triangulum-10b-f16.gguf
```

### Step 2: Model Initialization and Download

Upon running the command, Ollama will initialize and download the model files. You should see output similar to the following:

```plaintext
pulling manifest
pulling 8934d96d3f08... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████▏ 3.8 GB
pulling 8c17c2ebb0ea... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████▏ 7.0 KB
pulling 7c23fb36d801... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████▏ 4.8 KB
pulling 2e0493f67d0c... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████▏   59 B
pulling fa304d675061... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████▏   91 B
pulling 42ba7f8a01dd... 100% ▕██████████████████████████████████████████████████████████████████████████████████████████▏  557 B
verifying sha256 digest
writing manifest
removing any unused layers
success
>>> Send a message (/? for help)
```

### Step 3: Interact with the Model

Once the model is ready, you can send a prompt. For example, let's ask:

```plaintext
>>> What can you do for me?
```

If you asked the same question, you should receive a response like this:

```plaintext
As a responsible AI language model, I am here to assist you with any questions or tasks you may have. Here are some examples of things I can help with:

1. Answering questions: I can provide information on a wide range of topics, from science and technology to history and culture.
2. Generating ideas: I can help you brainstorm ideas for creative projects, or provide suggestions for solving problems.
3. Writing assistance: I can help you with writing tasks such as proofreading, editing, and suggesting alternative words or phrases.
4. Translation: I can translate text from one language to another.
5. Summarizing content: I can summarize long pieces of text, such as articles or documents, into shorter, more digestible versions.
6. Creativity: I can help you generate creative ideas for stories, poems, or other forms of writing.
7. Language learning: I can assist you in learning a new language by providing grammar explanations, vocabulary lists, and practice exercises.
8. Chatting: I'm here to chat with you and provide a response to any question or topic you'd like to discuss.

Please let me know if there is anything specific you would like me to help you with.
```

### Step 4: Exit the Program

To exit the program, simply type:

```plaintext
/exit
```

## Example 2: Running Multi-Modal Models (Future Use)

Ollama supports running multi-modal models where you can send images and ask questions based on them. This section will be updated as more models become available.

## Notes on Using Quantized Models

Quantized models like **triangulum-10b-f16.gguf** are optimized for performance on resource-constrained hardware, making it accessible for local inference.

1. Ensure your system has sufficient VRAM or CPU resources.
2. Use the `.gguf` model format for compatibility with Ollama.

# **Conclusion**

Running the **Triangulum-10B** model with Ollama provides a robust way to leverage open-source LLMs locally for diverse use cases. By following these steps, you can explore the capabilities of other open-source models in the future.