# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# improved from coconut_stabilized.py to include parallelism

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "stats"], defaults=(None,)
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
            self.adapter_gate = nn.Parameter(torch.zeros(1))
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
        stats = {
            "token_embed_rms": float(token_rms_tensor.cpu()),
        }

        if self.adapter_type == "scale":
            hidden_rms_tensor = (
                hidden_vec.detach().float().norm(p=2) / math.sqrt(hidden_dim)
            )
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


class CoconutParallel(nn.Module):
    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        *,
        latent_detach=True,
        latent_adapter="ln",
        parallel_mode="triangular",
        num_parallel_passes=None,
        ema_decay=0.0,
        latent_init="token",
        latent_init_std=0.02,
        tc_weight=0.05,
        sd_weight=0.0,
        tc_distance="mse",
        tc_norm="rms",
        use_tail_only_forward=True,
        use_prefix_kv_cache=True,
    ):

        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.latent_detach = latent_detach
        self.latent_adapter_type = latent_adapter.lower()
        self.parallel_mode = parallel_mode.lower()
        self.num_parallel_passes = num_parallel_passes
        self.ema_decay = ema_decay
        self.latent_init = latent_init.lower()
        self.latent_init_std = latent_init_std
        self.tc_weight = tc_weight
        self.sd_weight = sd_weight
        self.tc_distance = tc_distance.lower()
        self.tc_norm = tc_norm.lower()
        self.use_tail_only_forward = bool(use_tail_only_forward)
        self.use_prefix_kv_cache = bool(use_prefix_kv_cache)
        self._norm_eps = 1e-6

        if self.parallel_mode not in {"triangular", "full"}:
            raise ValueError(
                f"Unknown parallel_mode '{parallel_mode}'. Choose 'triangular' or 'full'."
            )
        if self.tc_distance not in {"mse", "cos"}:
            raise ValueError(
                f"Unknown tc_distance '{tc_distance}'. Choose 'mse' or 'cos'."
            )
        if self.tc_norm not in {"rms", "none"}:
            raise ValueError(
                f"Unknown tc_norm '{tc_norm}'. Choose 'rms' or 'none'."
            )
        if not (0.0 <= self.ema_decay <= 1.0):
            raise ValueError("ema_decay must be in [0, 1].")

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

        self.latent_adapter = LatentAdapter(
            self.embedding, self.latent_token_id, self.latent_adapter_type
        )

        hidden_dim = self.embedding.embedding_dim
        if self.latent_init == "learned_slot":
            self.latent_slot_params = nn.Parameter(
                torch.zeros(MAX_N_LATENT, hidden_dim)
            )
            nn.init.normal_(self.latent_slot_params, mean=0.0, std=self.latent_init_std)
        else:
            self.register_parameter("latent_slot_params", None)

        self.parallel_inference = True
        self.past_kv_prefix = None
        self.cached_upto = 0

    def _prepare_latent_lists(self, input_ids):
        latent_indices = (input_ids == self.latent_token_id).nonzero(as_tuple=False)
        batch_size = input_ids.shape[0]
        latent_lists = [[] for _ in range(batch_size)]
        for batch_idx, token_idx in latent_indices.tolist():
            latent_lists[batch_idx].append(token_idx)

        max_n_latents = max((len(lst) for lst in latent_lists), default=0)
        exists_mask = None
        if max_n_latents > 0:
            device = input_ids.device
            exists_mask = torch.zeros(
                batch_size, max_n_latents, dtype=torch.bool, device=device
            )
            for batch_idx, lst in enumerate(latent_lists):
                if lst:
                    exists_mask[batch_idx, : len(lst)] = True

        return latent_lists, max_n_latents, exists_mask

    def _initialize_latents(self, inputs_embeds, latent_lists):
        if not latent_lists or all(len(lst) == 0 for lst in latent_lists):
            return inputs_embeds

        if self.latent_init == "token":
            return inputs_embeds

        inputs_embeds = inputs_embeds.clone()
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        for batch_idx, positions in enumerate(latent_lists):
            if not positions:
                continue
            index_tensor = torch.tensor(positions, device=device, dtype=torch.long)

            if self.latent_init == "learned_slot":
                if len(positions) > MAX_N_LATENT:
                    raise ValueError(
                        f"learned_slot supports at most {MAX_N_LATENT} latents but got {len(positions)}"
                    )
                values = self.latent_slot_params[: len(positions)].to(device=device, dtype=dtype)
            elif self.latent_init == "gaussian":
                hidden_dim = inputs_embeds.shape[-1]
                values = torch.randn(
                    len(positions), hidden_dim, device=device, dtype=dtype
                )
                values = values * self.latent_init_std
            else:
                raise ValueError(
                    f"Unknown latent_init '{self.latent_init}'. Expected 'token', 'learned_slot', or 'gaussian'."
                )

            inputs_embeds[batch_idx].index_copy_(0, index_tensor, values)

        return inputs_embeds

    def _apply_tc_norm(self, tensor):
        if self.tc_norm == "none":
            return tensor
        denom = torch.sqrt(
            tensor.pow(2).mean(dim=-1, keepdim=True) + self._norm_eps
        )
        return tensor / denom

    def _distance(self, current, target):
        if self.tc_distance == "mse":
            return ((current - target) ** 2).mean(dim=-1)
        cos_sim = F.cosine_similarity(current, target, dim=-1)
        return 1.0 - cos_sim

    def _parallel_refine(
        self,
        input_ids,
        attention_mask,
        position_ids,
        inputs_embeds,
        latent_lists,
        max_n_latents,
        exists_mask,
        *,
        parallel_mode=None,
        num_parallel_passes=None,
        ema_decay=None,
        track_losses=True,
    ):

        device = input_ids.device
        dtype = inputs_embeds.dtype
        batch_size, _, hidden_dim = inputs_embeds.shape

        tail_start = 0
        prefix_len = 0
        use_tail = False

        if self.use_tail_only_forward and max_n_latents > 0:
            first_positions = [positions[0] for positions in latent_lists if positions]
            if first_positions:
                earliest_latent = min(first_positions)
                tail_start = max(0, earliest_latent - 1)
                prefix_len = tail_start
                use_tail = tail_start == 0 or self.use_prefix_kv_cache

        if not use_tail:
            tail_start = 0
            prefix_len = 0
            self.past_kv_prefix = None
            self.cached_upto = 0

        stats_accumulator = {
            "latent_gate": [],
            "latent_rms": [],
            "token_embed_rms": [],
            "updated_slots": [],
            "frozen_slots": [],
        }

        if max_n_latents == 0:
            self.gen_forward_cnt = 1
            zero = inputs_embeds.new_zeros(())
            return inputs_embeds, zero, zero, stats_accumulator, 0

        mode = (parallel_mode or self.parallel_mode).lower()
        if mode not in {"triangular", "full"}:
            raise ValueError(f"Unsupported parallel_mode '{mode}'")

        num_passes_cfg = (
            num_parallel_passes
            if num_parallel_passes is not None
            else self.num_parallel_passes
        )
        if num_passes_cfg is None:
            total_passes = max_n_latents
        else:
            total_passes = max(1, min(num_passes_cfg, max_n_latents))

        ema = self.ema_decay if ema_decay is None else ema_decay

        # tensors for consistency objectives
        loss_tc = inputs_embeds.new_zeros(())
        loss_sd = inputs_embeds.new_zeros(())
        pass_records = []
        latents_prev = None

        for pass_idx in range(total_passes):
            if use_tail:
                if (
                    self.use_prefix_kv_cache
                    and prefix_len > 0
                    and self.cached_upto < prefix_len
                ):
                    start = self.cached_upto
                    end = prefix_len
                    if end > start:
                        with torch.no_grad():
                            prefix_outputs = self.base_causallm(
                                inputs_embeds=inputs_embeds[:, start:end, :],
                                attention_mask=attention_mask[:, :end],
                                position_ids=position_ids[:, start:end],
                                past_key_values=self.past_kv_prefix,
                                use_cache=True,
                            )
                        prefix_past = prefix_outputs.past_key_values
                        self.past_kv_prefix = tuple(
                            tuple(tensor.detach() for tensor in layer)
                            for layer in prefix_past
                        )
                        self.cached_upto = end

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, tail_start:, :],
                    attention_mask=attention_mask,
                    position_ids=position_ids[:, tail_start:],
                    past_key_values=(
                        self.past_kv_prefix
                        if (self.use_prefix_kv_cache and prefix_len > 0)
                        else None
                    ),
                    output_hidden_states=True,
                    use_cache=self.use_prefix_kv_cache and prefix_len > 0,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states[-1]

            if track_losses:
                if latents_prev is None:
                    latents_current = torch.zeros(
                        batch_size,
                        max_n_latents,
                        hidden_dim,
                        device=device,
                        dtype=dtype,
                    )
                else:
                    latents_current = latents_prev.clone()

            updated_mask = torch.zeros(
                batch_size, max_n_latents, dtype=torch.bool, device=device
            )

            updated_slots_this_pass = 0
            frozen_slots_this_pass = 0

            for batch_idx, positions in enumerate(latent_lists):
                if not positions:
                    continue

                num_slots = len(positions)
                if mode == "triangular":
                    start_slot = min(pass_idx, num_slots)
                else:
                    start_slot = 0

                if mode == "triangular":
                    frozen_slots_this_pass += min(start_slot, num_slots)

                if start_slot >= num_slots:
                    continue

                update_positions = positions[start_slot:]
                update_values = []

                for local_offset, pos in enumerate(
                    update_positions, start=start_slot
                ):
                    offset = tail_start if use_tail else 0
                    source_index = pos - 1 - offset
                    if source_index < 0:
                        continue
                    if source_index >= hidden_states.shape[1]:
                        continue

                    state = hidden_states[batch_idx, source_index, :]
                    if self.latent_detach:
                        state = state.detach()

                    adapted_state, adapter_stats = self.latent_adapter(state)
                    prev_embed = inputs_embeds[batch_idx, pos, :]
                    if ema > 0.0:
                        adapted_state = (1.0 - ema) * adapted_state + ema * prev_embed.detach()

                    update_values.append(adapted_state)

                    if track_losses:
                        latents_current[batch_idx, local_offset, :] = adapted_state
                    updated_mask[batch_idx, local_offset] = True
                    updated_slots_this_pass += 1

                    for key, value in adapter_stats.items():
                        if value is not None:
                            stats_accumulator.setdefault(key, []).append(value)

                if update_values:
                    index_tensor = torch.tensor(
                        update_positions, device=device, dtype=torch.long
                    )
                    value_tensor = torch.stack(update_values, dim=0)
                    inputs_embeds[batch_idx].index_copy_(0, index_tensor, value_tensor)

            stats_accumulator["updated_slots"].append(updated_slots_this_pass)
            stats_accumulator["frozen_slots"].append(frozen_slots_this_pass)

            if track_losses:
                if latents_prev is not None:
                    valid_mask = updated_mask
                    if exists_mask is not None:
                        valid_mask = valid_mask & exists_mask

                    if valid_mask.any():
                        current_norm = self._apply_tc_norm(latents_current)[valid_mask]
                        prev_norm = self._apply_tc_norm(latents_prev.detach())[valid_mask]
                        loss_tc = loss_tc + self._distance(
                            current_norm, prev_norm
                        ).sum()

                latents_prev = latents_current
                pass_records.append(
                    {
                        "latents": latents_current,
                        "updated_mask": updated_mask,
                    }
                )

        self.gen_forward_cnt = total_passes + 1

        if track_losses and len(pass_records) > 1:
            final_latents = pass_records[-1]["latents"].detach()
            final_norm = self._apply_tc_norm(final_latents)
            for record in pass_records[:-1]:
                mask = record["updated_mask"]
                if exists_mask is not None:
                    mask = mask & exists_mask
                if mask.any():
                    curr = self._apply_tc_norm(record["latents"])[mask]
                    target = final_norm[mask]
                    loss_sd = loss_sd + self._distance(curr, target).sum()

        if self.use_prefix_kv_cache:
            self.past_kv_prefix = None
            self.cached_upto = 0

        return inputs_embeds, loss_tc, loss_sd, stats_accumulator, total_passes

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        self.past_kv_prefix = None
        self.cached_upto = 0
        inputs_embeds = self.embedding(input_ids)
        latent_lists, max_n_latents, exists_mask = self._prepare_latent_lists(
            input_ids
        )
        inputs_embeds = self._initialize_latents(inputs_embeds, latent_lists)

        (
            refined_embeds,
            loss_tc,
            loss_sd,
            stats_accumulator,
            total_passes,
        ) = self._parallel_refine(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            latent_lists,
            max_n_latents,
            exists_mask,
            track_losses=True,
        )

        outputs = self.base_causallm(
            inputs_embeds=refined_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss_nll = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        total_loss = loss_nll + self.tc_weight * loss_tc + self.sd_weight * loss_sd

        aggregated_stats = {}
        for key, values in stats_accumulator.items():
            if values:
                aggregated_stats[key] = float(sum(values) / len(values))
            else:
                aggregated_stats[key] = None

        aggregated_stats["tc_loss"] = float(loss_tc.detach())
        aggregated_stats["sd_loss"] = float(loss_sd.detach())
        aggregated_stats["gen_forward_cnt"] = total_passes + 1

        return Outputs(
            loss=total_loss,
            inputs_embeds=refined_embeds,
            logits=logits,
            stats=aggregated_stats,
        )

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        parallel_mode=None,
        num_parallel_passes=None,
        ema_decay=None,
        **kwargs,
    ):

        self.gen_forward_cnt = 0
        self.past_kv_prefix = None
        self.cached_upto = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        if not getattr(self, "parallel_inference", True):
            attn = attention_mask
            if attn is None:
                attn = torch.ones_like(input_ids, device=input_ids.device)
            base_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attn,
                "max_new_tokens": max_new_tokens,
            }
            base_kwargs.update(
                {
                    key: value
                    for key, value in kwargs.items()
                    if key
                    not in {
                        "parallel_mode",
                        "num_parallel_passes",
                        "ema_decay",
                        "position_ids",
                    }
                }
            )
            return self.base_causallm.generate(**base_kwargs)

        device = input_ids.device
        tokens = input_ids[0].detach().tolist()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=device
            ).unsqueeze(0)

        with torch.no_grad():
            embeds = self.embedding(input_ids)
            latent_lists, max_n_latents, exists_mask = self._prepare_latent_lists(
                input_ids
            )
            embeds = self._initialize_latents(embeds, latent_lists)

            (
                refined_embeds,
                _,
                _,
                _,
                total_passes,
            ) = self._parallel_refine(
                input_ids,
                attention_mask,
                position_ids,
                embeds,
                latent_lists,
                max_n_latents,
                exists_mask,
                parallel_mode=parallel_mode,
                num_parallel_passes=num_parallel_passes,
                ema_decay=ema_decay,
                track_losses=False,
            )

            outputs = self.base_causallm(
                inputs_embeds=refined_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            logits = outputs.logits
            next_token = torch.argmax(logits[0, -1]).item()
            tokens.append(next_token)

            next_token_tensor = torch.tensor(
                [[next_token]], device=device, dtype=torch.long
            )
            next_token_embed = self.embedding(next_token_tensor).view(1, 1, -1)

            refined_embeds = torch.cat((refined_embeds, next_token_embed), dim=1)
            input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, 1))), dim=1
            )

            self.gen_forward_cnt = total_passes + 1

            for _ in range(max_new_tokens - 1):
                position_ids = torch.arange(
                    0, refined_embeds.shape[1], dtype=torch.long, device=device
                ).unsqueeze(0)

                outputs = self.base_causallm(
                    inputs_embeds=refined_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                self.gen_forward_cnt += 1

                next_token = torch.argmax(outputs.logits[0, -1]).item()
                if next_token == self.eos_token_id:
                    break
                tokens.append(next_token)

                next_token_tensor = torch.tensor(
                    [[next_token]], device=device, dtype=torch.long
                )
                next_token_embed = self.embedding(next_token_tensor).view(1, 1, -1)
                refined_embeds = torch.cat((refined_embeds, next_token_embed), dim=1)
                input_ids = torch.cat((input_ids, next_token_tensor), dim=1)
                attention_mask = torch.cat(
                    (attention_mask, attention_mask.new_ones((1, 1))), dim=1
                )

            if synced_gpus:
                while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                    self.gen_forward_cnt += 1
                    _ = self.base_causallm(
                        inputs_embeds=refined_embeds,
                        attention_mask=attention_mask,
                        position_ids=torch.arange(
                            0, refined_embeds.shape[1], dtype=torch.long, device=device
                        ).unsqueeze(0),
                    )

            if output_embedding:
                return torch.tensor(tokens, device=device).view(1, -1), refined_embeds

            return torch.tensor(tokens, device=device).view(1, -1)
