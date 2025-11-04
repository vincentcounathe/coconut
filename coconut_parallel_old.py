import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

# Match coconut.py's output contract
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

class ParallelCoconut(nn.Module):
    """
    Parallel Coconut: K refinement passes over N latent slots in-place.

    Key idea:
      - Replace <|latent|> token *inputs* with a "slot tape" (vectors).
      - Run the model over the full sequence K times; after each pass,
        overwrite the slot vectors with the last-layer hidden states at those positions.
      - Final decode pass computes logits/loss (masked outside answer tokens via collator).

    Notes:
      - No teacher-consistency loss here (placeholder only). Can add it later by
        comparing AR Coconut b_i vs slot vectors v^k_i with.
    """

    def __init__(
        self,
        base_causallm: nn.Module,
        latent_token_id: int,
        start_latent_id: int,
        end_latent_id: int,
        eos_token_id: int,
        num_refine: int = 2,                # K
        slot_init: str = "random",          # {"random", "fixed", "learned"}
        slot_dim: int = None,               # inferred from embedding if None
        learned_slot_count: int = 64,       # only used if slot_init == "learned"
        projector_dim: int = 0,             # 0 = no projector; else add small head for future TC loss
    ):
        super().__init__()
        self.base = base_causallm
        self.eos_token_id = eos_token_id
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.num_refine = num_refine
        self.slot_init = slot_init
        self.learned_slot_count = learned_slot_count

        # Handle embeddings across model families (GPT2 vs others), same as coconut.py
        if isinstance(self.base, GPT2LMHeadModel):
            self.embedding = self.base.transformer.get_input_embeddings()
        else:
            self.embedding = self.base.get_input_embeddings()

        self.hidden_size = self.embedding.embedding_dim if hasattr(self.embedding, "embedding_dim") else self.embedding.weight.shape[1]
        self.slot_dim = self.hidden_size if slot_dim is None else slot_dim

        # Slot init parameters
        if slot_init == "fixed":
            # One trainable vector copied to all latent positions
            self.fixed_slot = nn.Parameter(torch.zeros(self.slot_dim))
            nn.init.normal_(self.fixed_slot, mean=0.0, std=0.02)
        elif slot_init == "learned":
            # A bank of learned vectors; we’ll map each latent position to an index via modulo
            self.slot_bank = nn.Parameter(torch.zeros(self.learned_slot_count, self.slot_dim))
            nn.init.normal_(self.slot_bank, mean=0.0, std=0.02)
        else:
            # "random": no parameters — we sample on-the-fly per batch
            self.register_parameter("fixed_slot", None)
            self.register_parameter("slot_bank", None)

        # Optional projector head for future thought-consistency (TC) loss (kept off by default)
        self.projector = None
        if projector_dim and projector_dim > 0:
            self.projector = nn.Sequential(
                nn.Linear(self.hidden_size, projector_dim, bias=False),
                nn.Tanh(),
                nn.Linear(projector_dim, self.hidden_size, bias=False),
            )

    # Mirror .train() / .eval() behavior of coconut.py
    def train(self, mode: bool = True):
        self.base.train(mode)
        super().train(mode)

    def eval(self):
        self.base.eval()
        super().eval()

    @torch.no_grad()
    def _init_slots(self, inputs_embeds: torch.Tensor, latent_mask: torch.Tensor):
        """
        Initialize slots for latent positions per instance.

        inputs_embeds: (bsz, seqlen, hidden)
        latent_mask:   (bsz, seqlen) bool, True where token == <|latent|>

        Returns:
            slots: (bsz, seqlen, hidden) — all zeros except latent positions filled
                  OR a separate structure (indices + vectors); here we return full tensor
                  to simplify scatter/gather ops.
        """
        bsz, seqlen, hidden = inputs_embeds.shape
        device = inputs_embeds.device
        slots = torch.zeros_like(inputs_embeds)  # alloc

        if self.slot_init == "fixed":
            # Broadcast a single learned vector
            slots[latent_mask] = self.fixed_slot.to(device)

        elif self.slot_init == "learned":
            # Deterministic mapping from position index -> a slot from the bank (by modulo)
            # Build a per-position index map
            pos_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
            bank_idx = (pos_ids % self.learned_slot_count)[latent_mask]  # (num_latent_total,)
            slots_flat = slots.view(-1, hidden)
            latent_flat_idx = latent_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)
            slots_flat[latent_flat_idx] = self.slot_bank[bank_idx].to(device)

        else:
            # "random": sample per batch per position
            num_latent = latent_mask.sum()
            if num_latent > 0:
                slots[latent_mask] = torch.randn(int(num_latent), hidden, device=device) * 0.02     # modify if we change base model

        return slots

    def _replace_latents_in_inputs(self, inputs_embeds: torch.Tensor, slots: torch.Tensor, latent_mask: torch.Tensor):
        """
        Replace token embeddings at latent positions with slot vectors.
        """
        if latent_mask.any():
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[latent_mask] = slots[latent_mask]
        return inputs_embeds

    def forward(self, input_ids, attention_mask, labels, position_ids=None, **kwargs):
        """
        K refinement passes over latent positions, then final decode pass for logits/loss.

        Inputs match coconut.Coconut.forward and the trainer loop expectations.
        """
        device = input_ids.device
        bsz, seqlen = input_ids.shape

        # Build initial input embeddings from tokens
        inputs_embeds = self.embedding(input_ids)

        # Identify latent positions (<|latent|>)
        latent_mask = (input_ids == self.latent_token_id)  # (bsz, seqlen) bool

        # Initialize slots
        slots = self._init_slots(inputs_embeds, latent_mask)

        # Optional: projector placeholder (for future TC loss)
        def maybe_project(x):
            return self.projector(x) if (self.projector is not None) else x

        # === K refinement passes ===
        # Note on update policy:
        # - we don't progressively "commit" early latent thoughts (e.g., fix b_1 after pass 1,
        #   then refine the remaining slots only).
        # - we update all latent slots on every pass — there is no freezing or
        #   commit of earlier slots. num_refine (K) is independent of the number of latent slots (N).
        for _ in range(self.num_refine):
            # Compose inputs: token embeds; replace latent positions with current slots
            refined_inputs = self._replace_latents_in_inputs(inputs_embeds, slots, latent_mask)

            # Run full forward; need last hidden states to update slots
            outputs = self.base(
                inputs_embeds=refined_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            hidden_last = outputs.hidden_states[-1]  # (bsz, seqlen, hidden)

            # Update slots at latent positions with the *current* last hidden state
            if latent_mask.any():
                slots = slots.clone()
                slots[latent_mask] = hidden_last[latent_mask]

            #.

        # === Final decode pass for logits/loss ===
        final_inputs = self._replace_latents_in_inputs(inputs_embeds, slots, latent_mask)
        outputs = self.base(
            inputs_embeds=final_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
        )

        logits = outputs.logits  # (bsz, seqlen, vocab)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return Outputs(loss=loss, inputs_embeds=final_inputs, logits=logits)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        position_ids=None,
        **kwargs
    ):
        """
        Greedy decode with K refinement first, then standard token decoding.

        Note: we follow the simplest path (no synced_gpus machinery); in DDP/FSDP eval, keep K the same across ranks.
        """
        self.eval()
        device = input_ids.device
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        # Initial embeddings and latent mask on the *prompt* portion
        inputs_embeds = self.embedding(input_ids)
        latent_mask = (input_ids == self.latent_token_id)

        # Init slots
        slots = self._init_slots(inputs_embeds, latent_mask)

        # K refinement passes on the prompt
        for _ in range(self.num_refine):
            refined_inputs = self._replace_latents_in_inputs(inputs_embeds, slots, latent_mask)
            outputs = self.base(inputs_embeds=refined_inputs, attention_mask=attention_mask, position_ids=position_ids, output_hidden_states=True)
            hidden_last = outputs.hidden_states[-1]
            if latent_mask.any():
                slots[latent_mask] = hidden_last[latent_mask]

        # Final compose before generation
        refined_inputs = self._replace_latents_in_inputs(inputs_embeds, slots, latent_mask)

        # First next token from the refined prompt
        outputs = self.base(inputs_embeds=refined_inputs, attention_mask=attention_mask, position_ids=position_ids)
        next_token = torch.argmax(outputs.logits[0, -1]).view(1, 1)

        tokens = torch.cat([input_ids, next_token.to(device)], dim=1)
        # Greedy loop
        for _ in range(max_new_tokens - 1):
            out = self.base(input_ids=tokens)
            nxt = torch.argmax(out.logits[0, -1]).item()
            if nxt == self.eos_token_id:
                break
            tokens = torch.cat([tokens, torch.tensor([[nxt]], device=device)], dim=1)

        if output_embedding:
            # Not returning embeddings here; keep parity with Coconut API.
            return tokens
        else:
            return tokens
