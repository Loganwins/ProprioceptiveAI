"""Map the Ollama model names people use to their Hugging Face equivalents (for deep mode).

Ollama does not expose hidden states, so deep view and Behaviour run the same model's Hugging Face weights.
This table lets Lens tell an Ollama user the exact command for the model they already use.
"""
from __future__ import annotations

OLLAMA_TO_HF = {
    # Qwen
    "qwen2.5:0.5b": "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct", "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder:0.5b": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "qwen2.5-coder:1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "qwen2.5-coder:3b": "Qwen/Qwen2.5-Coder-3B-Instruct", "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2:0.5b": "Qwen/Qwen2-0.5B-Instruct", "qwen2:7b": "Qwen/Qwen2-7B-Instruct",
    "qwen3:0.6b": "Qwen/Qwen3-0.6B", "qwen3:1.7b": "Qwen/Qwen3-1.7B", "qwen3:4b": "Qwen/Qwen3-4B", "qwen3:8b": "Qwen/Qwen3-8B",
    # Llama (ungated copies of Meta's weights, so no Hugging Face account is needed)
    "llama3.2:1b": "unsloth/Llama-3.2-1B-Instruct", "llama3.2:3b": "unsloth/Llama-3.2-3B-Instruct",
    "llama3.1:8b": "unsloth/Llama-3.1-8B-Instruct", "llama3:8b": "NousResearch/Meta-Llama-3-8B-Instruct",
    "hermes3:8b": "NousResearch/Hermes-3-Llama-3.1-8B", "tinyllama:1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # Mistral, Gemma
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3", "mistral-nemo:12b": "m8than/Mistral-Nemo-Instruct-2407-lenient-chatfix",
    "gemma2:2b": "Efficient-Large-Model/gemma-2-2b-it", "gemma3:1b": "unsloth/gemma-3-1b-it",
    # Microsoft, DeepSeek
    "phi3:3.8b": "microsoft/Phi-3-mini-4k-instruct", "phi3.5:3.8b": "microsoft/Phi-3.5-mini-instruct",
    "phi4-mini:3.8b": "microsoft/Phi-4-mini-instruct", "phi4-mini-reasoning:3.8b": "microsoft/Phi-4-mini-reasoning",
    "deepseek-r1:1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "deepseek-coder:1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-coder:6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    # Others
    "smollm2:1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2:360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm3:3b": "HuggingFaceTB/SmolLM3-3B", "granite3.3:2b": "ibm-granite/granite-3.3-2b-instruct",
    "granite3.3:8b": "ibm-granite/granite-3.3-8b-instruct", "granite3.2:8b": "ibm-granite/granite-3.2-8b-instruct",
    "granite3.1-dense:2b": "ibm-granite/granite-3.1-2b-instruct", "granite3.1-dense:8b": "ibm-granite/granite-3.1-8b-instruct",
    "falcon3:3b": "tiiuae/Falcon3-3B-Instruct", "falcon3:7b": "tiiuae/Falcon3-7B-Instruct",
    "olmo2:1b": "allenai/OLMo-2-0425-1B-Instruct", "olmo2:7b": "allenai/OLMo-2-1124-7B-Instruct",
    "stablelm-zephyr:3b": "stabilityai/stablelm-zephyr-3b",   # (EXAONE, InternLM2: need bundled model code; not supported)
    "qwen2:1.5b": "Qwen/Qwen2-1.5B-Instruct", "qwen:0.5b": "Qwen/Qwen1.5-0.5B-Chat", "qwen:1.8b": "Qwen/Qwen1.5-1.8B-Chat",
    "zephyr:7b": "HuggingFaceH4/zephyr-7b-beta", "openchat:7b": "openchat/openchat-3.5-1210",
    "solar:10.7b": "upstage/SOLAR-10.7B-Instruct-v1.0", "llama2:7b": "unsloth/llama-2-7b-chat",
    "llama2:13b": "Featherless-Chat-Models/llama2-13b-chat-hf", "falcon:7b": "tiiuae/falcon-7b-instruct",
    "foundation-sec:8b": "fdtn-ai/Foundation-Sec-8B-Instruct", "danube3:500m": "h2oai/h2o-danube3-500m-chat",
}
# More specific Ollama tags that point at a different checkpoint than the base name.
TAGGED = {
    "qwen3:4b-instruct-2507": "Qwen/Qwen3-4B-Instruct-2507", "qwen3:4b-thinking-2507": "Qwen/Qwen3-4B-Thinking-2507",
    "deepseek-r1:8b-llama-distill": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "phi3:3.8b-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
    "mistral:7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3", "mistral:7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
    "deepseek-coder:6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
}
# What Ollama pulls when no tag (or ":latest") is given.
DEFAULT_SIZE = {
    "qwen3": "8b", "qwen2.5": "7b", "qwen2.5-coder": "7b", "qwen2": "7b", "llama3.2": "3b", "llama3.1": "8b", "llama3": "8b",
    "llama2": "7b", "mistral": "7b", "phi3": "3.8b", "phi3.5": "3.8b", "phi4-mini": "3.8b", "deepseek-r1": "8b",
    "smollm2": "1.7b", "granite3.3": "8b", "falcon3": "7b", "olmo2": "7b", "tinyllama": "1.1b", "zephyr": "7b",
    "openchat": "7b", "solar": "10.7b", "hermes3": "8b",
}


def hf_for(ollama_name: str) -> str | None:
    """Best Hugging Face match for an Ollama model name ('qwen2.5:3b-instruct-q4_K_M' -> Qwen/Qwen2.5-3B-Instruct)."""
    name = ollama_name.lower().split("/")[-1]
    base, _, tag = name.partition(":")
    for full, hf in TAGGED.items():
        if name == full or name.startswith(full + "-"):
            return hf
    size = tag.split("-")[0] if tag and tag != "latest" else DEFAULT_SIZE.get(base, "")
    return OLLAMA_TO_HF.get(f"{base}:{size}")
