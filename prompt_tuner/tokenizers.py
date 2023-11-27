from transformers import AutoTokenizer, AutoTokenizerFast
from prompt_tuner.soft_prompt import SoftPrompt

class LlamaSPTokenizerFast(AutoTokenizerFast):
    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_prefix_space=False,
        **kwargs
    ):
    super().__init__(
        vocab_file,
        tokenizer_file=tokenizer_file,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )
    SoftPrompt._register_tokenizer(self)

class LlamaSPTokenizer(AutoTokenizer):
    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_prefix_space=False,
        **kwargs
    ):
    super().__init__(
        vocab_file,
        tokenizer_file=tokenizer_file,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )
    SoftPrompt._register_tokenizer(self)


class MistralSPTokenizerFast(AutoTokenizerFast):
    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_prefix_space=False,
        **kwargs
    ):
    super().__init__(
        vocab_file,
        tokenizer_file=tokenizer_file,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )
    SoftPrompt._register_tokenizer(self)

class MistralSPTokenizer(AutoTokenizer):
    def __init__(
        self,
        vocab_file,
        tokenizer_file=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        add_prefix_space=False,
        **kwargs
    ):
    super().__init__(
        vocab_file,
        tokenizer_file=tokenizer_file,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        add_prefix_space=add_prefix_space,
        **kwargs,
    )
    SoftPrompt._register_tokenizer(self)
