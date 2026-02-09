"""
LLM loading and generation utilities.

This module provides functions to load and use the Qwen2.5-3B-Instruct model
for text generation.
"""

import os
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Default model configuration
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# Global model and tokenizer (lazy loaded)
_model = None
_tokenizer = None


def load_model(
    model_name: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the LLM model and tokenizer.

    Args:
        model_name: HuggingFace model name. Defaults to Qwen2.5-3B-Instruct.
        device_map: Device placement strategy. Defaults to "auto".
        torch_dtype: Model precision. Defaults to float16.

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer

    if model_name is None:
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

    if _model is None or _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        print(f"Model '{model_name}' loaded successfully.")

    return _model, _tokenizer


def get_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Get the loaded model and tokenizer, loading them if necessary.

    Returns:
        Tuple of (model, tokenizer)
    """
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        return load_model()
    return _model, _tokenizer


def generate_response(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate a response for a single prompt (single-turn).

    Args:
        prompt: The user's input text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        do_sample: Whether to use sampling.

    Returns:
        The generated response text.
    """
    model, tokenizer = get_model_and_tokenizer()

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample and temperature > 0:
        gen_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **gen_kwargs)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    return response


def generate_from_messages(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 50,
) -> str:
    """
    Generate a response from a list of chat messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature. 0 = deterministic.
        top_p: Nucleus sampling parameter.
        top_k: Top-k sampling parameter.

    Returns:
        The generated response text.
    """
    model, tokenizer = get_model_and_tokenizer()

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    do_sample = temperature > 0.0

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()

    return text
