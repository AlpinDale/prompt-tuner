from transformers import LlamaTokenizerFast
from prompt_tuner.tuning import LlamaPromptTuningLM
from prompt_tuner.soft_prompt import SoftPrompt
import pytest
from prompt_tuner.inference import LlamaSoftPromptLM
from prompt_tuner.tokenizers import LlamaSPTokenizerFast

inf_model = LlamaSoftPromptLM.from_pretrained("NousResearch/Llama-2-7b-hf")
inf_tokenizer = LlamaSPTokenizerFast.from_pretrained("NousResearch/Llama-2-7b-hf")

tun_model = LlamaPromptTuningLM.from_pretrained("NousResearch/Llama-2-7b-hf")
tun_tokenizer = LlamaTokenizerFast.from_pretrained("NousResearch/Llama-2-7b-hf")

@pytest.fixture(scope="session", autouse=True)
def inference_resources(request):
    return (inf_model, inf_tokenizer)

@pytest.fixture(scope="session", autouse=True)
def tuning_resources(request):
    tun_model.initialize_soft_prompt()
    return (tun_model, tun_tokenizer)