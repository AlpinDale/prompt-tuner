from prompt_tuner.inference import LlamaSoftPromptLM
from prompt_tuner.tokenizers import LlamaSPTokenizerFast
from prompt_tuner.soft_prompt import SoftPrompt

def test_instantiate_multiple_objects():
    # Act
    model_a = LlamaSoftPromptLM.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer_a = LlamaSPTokenizerFast.from_pretrained("NousResearch/Llama-2-7b-hf")

    sp_a = SoftPrompt.from_string("TEST",model=model_a, tokenizer=tokenizer_a)

    model_b = LlamaSoftPromptLM.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer_b = LlamaSPTokenizerFast.from_pretrained("NousResearch/Llama-2-7b-hf")

    sp_b = SoftPrompt.from_string("TEST",model=model_b, tokenizer=tokenizer_b)

    # Assert
    assert model_a in SoftPrompt._models
    assert model_b in SoftPrompt._models
    assert tokenizer_a in SoftPrompt._tokenizers
    assert tokenizer_b in SoftPrompt._tokenizers
    assert sp_a in SoftPrompt._soft_prompts
    assert sp_b in SoftPrompt._soft_prompts
    assert len(tokenizer_a) == len(tokenizer_b)

    # Teardown
    del model_a
    del model_b
    del tokenizer_a
    del tokenizer_b
