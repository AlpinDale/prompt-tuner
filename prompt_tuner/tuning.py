from transformers import AutoModelForCausalLM
from prompt_tuner.soft_prompt import SoftPrompt
import torch
import torch.nn as nn
from torch.nn import Embedding


class UniversalPromptTuningMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        model.__class__ = type(
            "_UniversalPromptTuning" + model.__class__.__name__, 
            (UniversalPromptTuningMixin, model.__class__), 
            {}
        )

        for param in model.parameters():
            param.requires_grad = False

        model.initialize_soft_prompt()

        return model

    def initialize_soft_prompt(self, n_tokens=20):
        self.learned_embedding = nn.parameter.Parameter(
            self.get_input_embeddings().weight[:n_tokens].clone().detach()
        )

    def set_soft_prompt_embeds(self, soft_prompt_embeds):
        self.learned_embedding = nn.parameter.Parameter(
            soft_prompt_embeds.clone().detach()
        )

    def set_soft_prompt(self, sp: SoftPrompt):
        self.learned_embedding = nn.parameter.Parameter(
            sp.get_inputs_embeds().clone().detach().squeeze(0)
        )

    def get_soft_params(self):
        return self.learned_embedding

    def prepare_inputs_for_generation(self, input_ids, past=None, *args, **kwargs):
        input_ids = input_ids.to(self.device)
        return super().prepare_inputs_for_generation(
            input_ids, None, *args, **kwargs
        )

    def _cat_learned_embedding_to_input(self, input_ids):
        inputs_embeds = self.get_input_embeddings()(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            ie = inputs_embeds.unsqueeze(0)
        else:
            ie = inputs_embeds

        learned_embedding = self.transformer.drop(self.learned_embedding)

        inputs_embeds = torch.cat(
            [learned_embedding.repeat(ie.size(0), 1, 1), ie], dim=1
        )

        return inputs_embeds

    def _extend_labels(self, labels):
        n_tokens = self.learned_embedding.shape[-2]

        if len(list(labels.shape)) == 1:
            lb = labels.unsqueeze(0)
        else:
            lb = labels

        # Add '-100's (prevent loss calculation where the learned embed would be)
        n_batches = lb.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), -100).to(self.device), lb], dim=1)

    def _extend_attention_mask(self, attention_mask):
        n_tokens = self.learned_embedding.shape[-2]

        if len(list(attention_mask.shape)) == 1:
            am = attention_mask.unsqueeze(0)
        else:
            am = attention_mask

        n_batches = am.shape[0]
        return torch.cat([torch.full((n_batches,n_tokens), 1).to(self.device), am], dim=1)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # This fixes CUDA for some reason
        kwargs['input_ids'] = kwargs['input_ids'].to(self.device)

        return super().generate(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        assert input_ids is not None
        assert input_ids.ndim == 2

        input_ids = torch.nn.functional.pad(input_ids, (self.learned_embedding.size(0), 0, 0, 0), value=self.get_input_embeddings().weight.size(0) // 2)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        old_embedding_call = Embedding.__call__
        model = self

        def new_embedding_call(self, input_ids, *args, **kwargs):
            inputs_embeds = old_embedding_call(self, input_ids, *args, **kwargs)
            if model.get_input_embeddings() is self:
                assert inputs_embeds.ndim == 3
                inputs_embeds[:, :model.learned_embedding.size(0), :] = model.learned_embedding[None]
            return inputs_embeds

        Embedding.__call__ = new_embedding_call

        try:
            # Drop most of the args for now
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
            )
        finally:
            Embedding.__call__ = old_embedding_call

class AutoPromptTuningLM(UniversalPromptTuningMixin, AutoModelForCausalLM):
    def __init__(self, config):
        super().__init__(config)
