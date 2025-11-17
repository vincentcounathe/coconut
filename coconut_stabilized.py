# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# improved from coconut.py to include stab tricks

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple(
    "Outputs",
    ["loss", "inputs_embeds", "logits", "stats", "past_key_values"],
    defaults=(None, None),
)
MAX_N_LATENT = 8


class LatentAdapter(nn.Module):
    def __init__(self, embedding_layer, latent_token_id, adapter_type="none"):
        super().__init__()
        self.embedding = embedding_layer
        self.latent_token_id = latent_token_id
        self.adapter_type = adapter_type.lower()
        hidden_dim = embedding_layer.embedding_dim

        if self.adapter_type == "ln":
            self.adapter_ln = nn.LayerNorm(hidden_dim)
            self.adapter_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            with torch.no_grad():
                self.adapter_linear.weight.copy_(
                    torch.eye(
                        hidden_dim, dtype=self.adapter_linear.weight.dtype
                    )
                )
            # init gate deep in sigmoid to force proj and LN (and no learned embed)
            self.adapter_gate = nn.Parameter(torch.full((1,), 1e5))
        elif self.adapter_type == "scale":
            self.register_buffer("_epsilon", torch.tensor(1e-6))
        elif self.adapter_type == "none":
            pass
        else:
            raise ValueError(f"Unknown latent_adapter option: {adapter_type}")

    def forward(self, hidden_vec):
        embed = self.embedding.weight[self.latent_token_id]
        hidden_dim = hidden_vec.shape[-1]

        token_rms_tensor = (
            embed.detach().float().norm(p=2) / math.sqrt(hidden_dim)
        )
        hidden_rms_tensor = (
            hidden_vec.detach().float().norm(p=2) / math.sqrt(hidden_dim)
        )
        stats = {
            "token_embed_rms": float(token_rms_tensor.cpu()),
            "pre_latent_rms": float(hidden_rms_tensor.cpu()),
        }

        if self.adapter_type == "scale":
            scale = token_rms_tensor / (hidden_rms_tensor + self._epsilon)
            scale = scale.to(hidden_vec.device, hidden_vec.dtype)
            adapted = hidden_vec * scale
            stats["latent_gate"] = scale.detach().cpu().item()

        elif self.adapter_type == "ln":
            normalized = self.adapter_ln(hidden_vec)
            projected = self.adapter_linear(normalized)
            gate = torch.sigmoid(self.adapter_gate)
            adapted = gate * projected + (1 - gate) * embed
            stats["latent_gate"] = gate.detach().cpu().item()

        else:
            adapted = hidden_vec
            stats["latent_gate"] = 1.0

        adapted_rms_tensor = (
            adapted.detach().float().norm(p=2) / math.sqrt(hidden_dim)
        )
        stats["latent_rms"] = float(adapted_rms_tensor.cpu())

        return adapted, stats


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        latent_detach=True,
        latent_adapter="none",
        log_attention_metrics=False,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.latent_detach = latent_detach
        self.latent_adapter_type = latent_adapter.lower()
        self.log_attention_metrics = log_attention_metrics

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        self.latent_adapter = LatentAdapter(
            self.embedding, self.latent_token_id, self.latent_adapter_type
        )

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []
        stats_accumulator = {
            'latent_gate': [],
            'latent_rms': [],
            'token_embed_rms': [],
            'pre_latent_rms': [],
            'prompt_embed_rms': [],
            'n_latent_tokens': [],
        }

        hidden_metric_names = [
            'h_first_latent_rms',
            'h_first_nonlatent_rms',
            'h_mid_latent_rms',
            'h_mid_nonlatent_rms',
            'h_last_latent_rms',
            'h_last_nonlatent_rms',
        ]
        for key in hidden_metric_names:
            stats_accumulator[key] = {'sum': 0.0, 'count': 0}

        attention_metric_names = []
        if self.log_attention_metrics:
            attention_metric_names = [
                'attn_to_latents_first',
                'attn_to_latents_mid',
                'attn_to_latents_last',
                'attn_from_latents_first',
                'attn_from_latents_mid',
                'attn_from_latents_last',
            ]
            for key in attention_metric_names:
                stats_accumulator[key] = {'sum': 0.0, 'count': 0}

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        latent_mask = input_ids == self.latent_token_id
        attention_mask_bool = attention_mask.bool()

        latent_indices = latent_mask.nonzero(as_tuple=False)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(batch_size)
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, seq_len)
        inputs_embeds = self.embedding(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(
            batch_size, -1
        )
        fill_value = torch.full_like(positions, seq_len)
        latent_positions = torch.where(latent_mask, positions, fill_value)
        first_latent_pos = latent_positions.min(dim=1).values
        has_latent = latent_mask.any(dim=1)
        first_latent_pos = torch.where(
            has_latent, first_latent_pos, torch.full_like(first_latent_pos, seq_len)
        )
        prompt_mask = positions < first_latent_pos.unsqueeze(1)
        prompt_mask = prompt_mask & attention_mask_bool

        latent_counts = latent_mask.sum(dim=1).float()
        stats_accumulator['n_latent_tokens'].append(
            float(latent_counts.mean().detach().cpu().item())
        )

        hidden_dim = inputs_embeds.shape[-1]
        token_rms = torch.sqrt(
            inputs_embeds.detach().float().pow(2).sum(dim=-1) / hidden_dim
        )
        prompt_values = token_rms[prompt_mask]
        if prompt_values.numel() > 0:
            stats_accumulator['prompt_embed_rms'].append(
                float(prompt_values.mean().cpu().item())
            )

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        def _update_hidden_state_stats(hidden_states_tuple, start_idx, end_idx):
            chunk_len = end_idx - start_idx
            if (
                chunk_len <= 0
                or hidden_states_tuple is None
                or len(hidden_states_tuple) == 0
            ):
                return

            hidden_layers = (
                list(hidden_states_tuple[1:])
                if len(hidden_states_tuple) > 1
                else list(hidden_states_tuple)
            )
            if not hidden_layers:
                return

            chunk_input_ids = input_ids[:, start_idx:end_idx]
            chunk_attention_mask = attention_mask_bool[:, start_idx:end_idx]
            if chunk_attention_mask.numel() == 0:
                return

            latent_chunk_mask = (
                (chunk_input_ids == self.latent_token_id) & chunk_attention_mask
            )
            nonlatent_chunk_mask = (
                (~(chunk_input_ids == self.latent_token_id)) & chunk_attention_mask
            )

            if not latent_chunk_mask.any() and not nonlatent_chunk_mask.any():
                return

            total_layers = len(hidden_layers)
            layer_indices = {
                'first': 0,
                'mid': total_layers // 2,
                'last': total_layers - 1,
            }
            hidden_dim_local = hidden_layers[0].shape[-1]

            for label, idx in layer_indices.items():
                layer_hidden = hidden_layers[idx]
                token_rms_layer = torch.sqrt(
                    layer_hidden.detach().float().pow(2).sum(dim=-1)
                    / hidden_dim_local
                )

                latent_values = token_rms_layer[latent_chunk_mask]
                if latent_values.numel() > 0:
                    stats_accumulator[f"h_{label}_latent_rms"]["sum"] += float(
                        latent_values.sum().detach().cpu().item()
                    )
                    stats_accumulator[f"h_{label}_latent_rms"]["count"] += int(
                        latent_values.numel()
                    )

                nonlatent_values = token_rms_layer[nonlatent_chunk_mask]
                if nonlatent_values.numel() > 0:
                    stats_accumulator[f"h_{label}_nonlatent_rms"]["sum"] += float(
                        nonlatent_values.sum().detach().cpu().item()
                    )
                    stats_accumulator[f"h_{label}_nonlatent_rms"]["count"] += int(
                        nonlatent_values.numel()
                    )

        def _update_attention_stats(attentions_tuple, start_idx, end_idx):
            if not self.log_attention_metrics or not attentions_tuple:
                return

            total_layers = len(attentions_tuple)
            if total_layers == 0:
                return

            layer_indices = {
                'first': 0,
                'mid': total_layers // 2,
                'last': total_layers - 1,
            }

            for label, idx in layer_indices.items():
                attn_tensor = attentions_tuple[idx]
                if attn_tensor is None:
                    continue

                attn_values = attn_tensor.detach().float()
                chunk_len = attn_values.shape[2]
                if chunk_len == 0:
                    continue
                context_len = attn_values.shape[-1]

                query_attention_mask = attention_mask_bool[
                    :, start_idx : start_idx + chunk_len
                ]
                query_latent_mask = latent_mask[:, start_idx : start_idx + chunk_len]

                key_attention_mask = attention_mask_bool[:, :context_len]
                key_latent_mask = latent_mask[:, :context_len] & key_attention_mask
                key_nonlatent_mask = (~latent_mask[:, :context_len]) & key_attention_mask

                key_latent_mask = key_latent_mask.unsqueeze(1).unsqueeze(2).to(
                    attn_values.dtype
                )
                key_nonlatent_mask = key_nonlatent_mask.unsqueeze(1).unsqueeze(2).to(
                    attn_values.dtype
                )

                per_query_latent_mass = (attn_values * key_latent_mask).sum(dim=-1)
                per_query_latent_mass = per_query_latent_mass.mean(dim=1)
                valid_queries = query_attention_mask.float()
                valid_count = valid_queries.sum()
                if valid_count.item() > 0:
                    stats_accumulator[f"attn_to_latents_{label}"]["sum"] += float(
                        (per_query_latent_mass * valid_queries)
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    )
                    stats_accumulator[f"attn_to_latents_{label}"]["count"] += float(
                        valid_count.detach().cpu().item()
                    )

                per_query_nonlatent_mass = (attn_values * key_nonlatent_mask).sum(
                    dim=-1
                )
                per_query_nonlatent_mass = per_query_nonlatent_mass.mean(dim=1)
                latent_query_mask = (query_latent_mask & query_attention_mask).float()
                latent_count = latent_query_mask.sum()
                if latent_count.item() > 0:
                    stats_accumulator[f"attn_from_latents_{label}"]["sum"] += float(
                        (per_query_nonlatent_mass * latent_query_mask)
                        .sum()
                        .detach()
                        .cpu()
                        .item()
                    )
                    stats_accumulator[f"attn_from_latents_{label}"]["count"] += float(
                        latent_count.detach().cpu().item()
                    )

        for pass_idx in range(max_n_latents):

            chunk_start, chunk_end = next_compute_range

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, chunk_start:chunk_end, :
                    ],
                    attention_mask=attention_mask[
                        :, chunk_start:chunk_end
                    ],
                    position_ids=position_ids[
                        :, chunk_start:chunk_end
                    ],
                    output_hidden_states=True,
                    output_attentions=self.log_attention_metrics,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : chunk_start, :],
                        v[:, :, : chunk_start, :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, chunk_start:chunk_end, :
                    ],
                    attention_mask=attention_mask[:, :chunk_end],
                    position_ids=position_ids[
                        :, chunk_start:chunk_end
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    output_attentions=self.log_attention_metrics,
                )

                hidden_states_offset = chunk_start
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)
            _update_hidden_state_stats(outputs.hidden_states, chunk_start, chunk_end)
            if self.log_attention_metrics:
                _update_attention_stats(outputs.attentions, chunk_start, chunk_end)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                state = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

                if self.latent_detach:
                    state = state.detach()

                adapted_state, adapter_stats = self.latent_adapter(state)
                tensor_list[batch_idx][token_idx] = adapted_state

                for key, value in adapter_stats.items():
                    if value is not None:
                        stats_accumulator.setdefault(key, []).append(value)

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        final_start, final_end = next_compute_range
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, final_start:final_end, :],
            attention_mask=attention_mask[:, :final_end],
            position_ids=position_ids[:, final_start:final_end],
            past_key_values=(
                [
                    (
                        k[:, :, : final_start, :],
                        v[:, :, : final_start, :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
            output_attentions=self.log_attention_metrics,
        )

        logits.append(outputs.logits)
        _update_hidden_state_stats(outputs.hidden_states, final_start, final_end)
        if self.log_attention_metrics:
            _update_attention_stats(outputs.attentions, final_start, final_end)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        supervised_mask = shift_labels.ne(-100)
        if supervised_mask.any():
            masked_logits = shift_logits[supervised_mask].detach().float()
            logit_rms_global = torch.sqrt(masked_logits.pow(2).mean())
            stats_accumulator.setdefault('logit_rms_global', []).append(
                float(logit_rms_global.cpu().item())
            )

            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            stats_accumulator.setdefault('prob_entropy_global', []).append(
                float(entropy.mean().cpu().item())
            )

        prev_tokens = input_ids[:, :-1]
        post_latent_mask = supervised_mask & (prev_tokens == self.latent_token_id)
        if post_latent_mask.any():
            latent_logits = shift_logits[post_latent_mask].detach().float()
            logit_rms_post_latent = torch.sqrt(latent_logits.pow(2).mean())
            stats_accumulator.setdefault('logit_rms_post_latent', []).append(
                float(logit_rms_post_latent.cpu().item())
            )

            log_probs_latent = F.log_softmax(latent_logits, dim=-1)
            probs_latent = log_probs_latent.exp()
            entropy_latent = -(probs_latent * log_probs_latent).sum(dim=-1)
            stats_accumulator.setdefault('prob_entropy_post_latent', []).append(
                float(entropy_latent.mean().cpu().item())
            )

        aggregated_stats = {}
        for key, values in stats_accumulator.items():
            if isinstance(values, list):
                aggregated_stats[key] = (
                    sum(values) / len(values) if values else None
                )
            elif isinstance(values, dict):
                aggregated_stats[key] = (
                    values['sum'] / values['count'] if values['count'] > 0 else None
                )
            else:
                aggregated_stats[key] = values

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            stats=aggregated_stats,
            past_key_values=outputs.past_key_values,
        )


    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()
        device = input_ids.device

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        kv_cache = outputs.past_key_values
        new_inputs_embeds = inputs_embeds
        attention_mask = torch.ones(
            (1, new_inputs_embeds.shape[1]), dtype=torch.long, device=device
        )

        next_token = torch.argmax(outputs.logits[0, -1]).item()
        generated = 0

        while generated < max_new_tokens and next_token != self.eos_token_id:
            tokens.append(next_token)
            token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=device
            )
            new_token_embed = self.embedding(token_tensor)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=device),
                ),
                dim=1,
            )
            generated += 1

            if generated == max_new_tokens:
                break

            if kv_cache is None:
                outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            else:
                position_id = torch.tensor(
                    [[attention_mask.shape[1] - 1]],
                    dtype=torch.long,
                    device=device,
                )
                outputs = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    attention_mask=attention_mask,
                    position_ids=position_id,
                    past_key_values=kv_cache,
                    use_cache=True,
                )

            self.gen_forward_cnt += 1
            kv_cache = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[0, -1]).item()

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)
