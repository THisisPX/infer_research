"""Real LLM weight and activation extraction via HuggingFace transformers.

Supports: LLaMA-2/3, Mistral, Mixtral (MoE), Qwen2-MoE.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerData:
    """Weight matrix and its collected activations for one linear layer."""
    name: str
    weight: torch.Tensor                  # (d_out, d_in) — note HF stores as (out, in)
    has_bias: bool = False
    bias: Optional[torch.Tensor] = None

    # Collected activations: list of (seq_len, d_in) tensors
    activations: List[torch.Tensor] = field(default_factory=list)

    # Layer type classification
    layer_type: str = "unknown"  # q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

    @property
    def d_in(self) -> int:
        return self.weight.shape[1]

    @property
    def d_out(self) -> int:
        return self.weight.shape[0]

    def clear_activations(self):
        self.activations = []


@dataclass
class ModelData:
    """Extracted weights and activations for a full model."""
    model_name: str
    layers: Dict[str, LayerData] = field(default_factory=dict)

    # Per-layer-type summary
    layer_types: Dict[str, List[str]] = field(default_factory=dict)


def extract_model_data(
    model,
    tokenizer,
    calibration_texts: List[str],
    max_seq_len: int = 2048,
    layers_to_hook: Optional[List[str]] = None,
    device: str = "cuda",
) -> ModelData:
    """Extract weights and collect activations from a HuggingFace model.

    Args:
        model: HuggingFace model (LLaMA, Mistral, Mixtral, etc.)
        tokenizer: corresponding tokenizer
        calibration_texts: list of calibration strings
        max_seq_len: max sequence length
        layers_to_hook: specific layer names to hook (None = all linear layers)
        device: device to run on

    Returns:
        ModelData with weights and collected activations
    """
    from transformers import PreTrainedModel
    if not isinstance(model, PreTrainedModel):
        raise TypeError("model must be a HuggingFace PreTrainedModel")

    model = model.to(device).eval()
    model_data = ModelData(model_name=model.config._name_or_path)

    # ── Step 1: Find all linear layers and extract weights ──
    linear_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if layers_to_hook is not None and name not in layers_to_hook:
                continue
            # Determine layer type
            if "q_proj" in name:
                ltype = "q_proj"
            elif "k_proj" in name:
                ltype = "k_proj"
            elif "v_proj" in name:
                ltype = "v_proj"
            elif "o_proj" in name:
                ltype = "o_proj"
            elif "gate_proj" in name or "wi" in name:
                ltype = "gate_proj"
            elif "up_proj" in name or "w1" in name:
                ltype = "up_proj"
            elif "down_proj" in name or "w2" in name:
                ltype = "down_proj"
            else:
                ltype = "linear"

            ld = LayerData(
                name=name,
                weight=module.weight.data.clone().cpu(),
                has_bias=module.bias is not None,
                bias=module.bias.data.clone().cpu() if module.bias is not None else None,
                layer_type=ltype,
            )
            model_data.layers[name] = ld
            linear_layers[name] = module

            if ltype not in model_data.layer_types:
                model_data.layer_types[ltype] = []
            model_data.layer_types[ltype].append(name)

    print(f"  Extracted {len(model_data.layers)} linear layers")
    for ltype, names in model_data.layer_types.items():
        print(f"    {ltype}: {len(names)} layers")

    # ── Step 2: Hook activations ──
    activations = {}

    def make_hook(layer_name):
        def hook_fn(module, inputs, output):
            # inputs[0]: (batch, seq, d_in) for linear
            # output: (batch, seq, d_out) for linear
            if isinstance(inputs, tuple):
                inp = inputs[0].detach().cpu()
            else:
                inp = inputs.detach().cpu()
            if layer_name not in activations:
                activations[layer_name] = []
            # Flatten batch: (batch*seq, d_in)
            activations[layer_name].append(inp.view(-1, inp.shape[-1]))
        return hook_fn

    hooks = []
    for name, module in linear_layers.items():
        hooks.append(module.register_forward_hook(make_hook(name)))

    # ── Step 3: Run calibration data ──
    print(f"  Running calibration on {len(calibration_texts)} texts...")
    total_tokens = 0
    with torch.no_grad():
        for i, text in enumerate(calibration_texts):
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_seq_len,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
            total_tokens += inputs["input_ids"].numel()
            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(calibration_texts)}] {total_tokens} tokens")
    print(f"  Calibration done: {total_tokens} total tokens")

    # Remove hooks
    for h in hooks:
        h.remove()

    # ── Step 4: Store activations ──
    for name, act_list in activations.items():
        if name in model_data.layers:
            # Concatenate all batches
            model_data.layers[name].activations = [torch.cat(act_list, dim=0)]

    # Count layers with activations
    n_with_act = sum(1 for ld in model_data.layers.values() if len(ld.activations) > 0)
    print(f"  Layers with activations: {n_with_act}/{len(model_data.layers)}")

    return model_data


# ── Quick loader for common models ───────────────────────────────────

def load_llama_8b(device: str = "cuda") -> tuple:
    """Load LLaMA-3-8B-Instruct."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    return model, tokenizer


def load_mistral_7b(device: str = "cuda") -> tuple:
    """Load Mistral-7B-v0.1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    return model, tokenizer


def load_mixtral_8x7b(device: str = "cuda") -> tuple:
    """Load Mixtral-8x7B-v0.1."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    return model, tokenizer


MODEL_LOADERS = {
    "llama-3-8b": load_llama_8b,
    "mistral-7b": load_mistral_7b,
    "mixtral-8x7b": load_mixtral_8x7b,
}


# ── WikiText-2 calibration ───────────────────────────────────────────

def get_wikitext_calibration(
    tokenizer,
    num_samples: int = 50,
    max_seq_len: int = 2048,
    seed: int = 42,
) -> List[str]:
    """Load WikiText-2 training texts for calibration."""
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter empty lines, join into blocks of reasonable length
    texts = [t for t in dataset["text"] if len(t.strip()) > 50]
    # Shuffle for diversity
    import random
    random.seed(seed)
    random.shuffle(texts)
    return texts[:num_samples]


def get_ptb_calibration(
    tokenizer,
    num_samples: int = 50,
    max_seq_len: int = 2048,
    seed: int = 42,
) -> List[str]:
    """Load Penn TreeBank texts for calibration."""
    from datasets import load_dataset
    dataset = load_dataset("ptb_text_only", split="train")
    texts = [t for t in dataset["sentence"] if len(t.strip()) > 50]
    import random
    random.seed(seed)
    random.shuffle(texts)
    return texts[:num_samples]
