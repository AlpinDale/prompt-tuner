# Soft Prompt Tuner
Soft Prompt Tuner is a library for training soft prompts for transformer causal language models.

Refer to the original paper for a model in-depth explanation: https://arxiv.org/abs/2104.08691


## What is a Soft Prompt?
Soft prompt modules are small binary files that can adjust the behaviour and textual biases of an LLM. The training process is similar to a standard finetune and/or LoRA tune.

In order for a transformer model to understand your prompt, it has to convert it to a format that its hidden layers can understand. To accomplish this, the most common character combinations (based on the model's pre-trained tokenizer) are mapped to token IDs before being fed to the model. The model then converts these token IDs to another internal format for the model's use via a process known as "embedding". The product of this process is a 2D array of "word embeddings" (or simply "embeddings").

Within the model's embedding matrix, there is one row for every possible token in the model's vocabulary (the full list of character-mapped token IDs). Embedding involves taking the rows from this 2D array that corresponds to the tokens in your story and concatenating them together.

Soft Prompts are 2D arrays that can be concatenated at the beginning of the embeddings directly after the embedding step (row-wise, so we are adding extra rows to the top of the embeddings, not extra columns to the left of the embeddings). This allows us to inject extra information to the model.

In essence, you can cram in a dataset's worth of information in a few dozen tokens' worth of context tokens in your prompt.

## Text Generation
```py
from prompt_tuner.inference import LlamaSoftPromptLM
from prompt_tuner.tokenizers import LlamaSPTokenizerFast
from prompt_tuner.soft_prompt import SoftPrompt
model = LlamaSoftPromptLM.from_pretrained("NousResearch/Llama-2-7b-hf")
tokenizer = LlamaSPTokenizerFast.from_pretrained("NousResearch/Llama-2-7b-hf")
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

sp = SoftPrompt.from_file("sample_sps/finetune/neuromancer_gpt2.json")
prompt = sp + "The sky over the port"
output = generator(prompt)
```
SoftPrompts can be concatenated at any point into your context as if they were strings. When the context is printed, SoftPrompts show up as human-readable tags for debugging. They also tokenize to the underlying number of tokens for easy budgeting.

See the [text generation notebook](text_generation.ipynb) for pointers on adding prompt_tuner to your generator.


## Training

For finetuneing soft prompts, check out the [finetune notebook](tuning_funetune.ipynb).

For AI text adventures or writing, the [World Info](tuning_world_info.ipynb) can be used.

## Limitations (for now)

- Still under testing.
- The Huggingface Trainer class should work as long as you set params=[model.get_soft_params()] on the optimizer, but it will still save full model checkpoints.
- prompt_tuner syncs a set of special tokens between its tokenizers the scenes. Adding your own tokens may result in unexpected behaviour.

## Credits
This work is forked off [corolla johnson](https://github.com/corolla-johnson/mkultra) and [VE FORBRYDERNE](https://github.com/VE-FORBRYDERNE/mkultra)'s works. 

Rest in peace, VE.
