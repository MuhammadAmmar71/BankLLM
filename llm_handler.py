import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import LLM_MODEL, MODE, MAX_NEW_TOKENS, REPETITION_PENALTY, SYSTEM_PROMPT, HF_TOKEN


def _set_hf_token():
    """Reads HF_TOKEN from env to enable authenticated HF Hub requests."""
    token = HF_TOKEN.strip() or os.environ.get("HF_TOKEN", "").strip()
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("[llm_handler] HF_TOKEN found — authenticated requests enabled.")
    else:
        print("[llm_handler] Warning: HF_TOKEN not set. "
              "Set it with: export HF_TOKEN=your_token_here")


#  Prompt templates 
def build_prompt(query: str, retrieved: list[dict]) -> str:
    """
    Builds the full prompt with retrieved context injected.
    Uses ChatML format for TinyLlama (dev) and Llama-3 format for prod.
    """
    context = "\n\n".join([
        f"[Source: {r['metadata']['sheet']}]\n{r['document']}"
        for r in retrieved
    ])

    if MODE == "dev":
        return (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n"
            f"Context from bank knowledge base:\n{context}\n\n"
            f"Customer Question: {query}</s>\n"
            f"<|assistant|>\n"
        )
    else:
        
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Context from bank knowledge base:\n{context}\n\n"
            f"Customer Question: {query}<|eot_id|>\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )


def extract_response(full_text: str) -> str:
    """Strips the prompt prefix and returns only the model's reply."""
    if MODE == "dev":
        return full_text.split("<|assistant|>")[-1].strip()
    else:
        return full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()


#  Model loading
def load_llm():
    """
    Loads the LLM tokenizer + model and returns a HuggingFace pipeline.
    - Clears max_length from the model's generation config to fix the
      'max_new_tokens vs max_length' conflict warning.
    - Generation params are passed at inference time, not pipeline creation,
      to avoid the GenerationConfig deprecation warning.
    """
    _set_hf_token()

    print(f"\n[llm_handler] Loading model : {LLM_MODEL}")
    print(f"              Mode          : {MODE.upper()}")
    print(f"              Device        : CPU")
    print("              (First load may take a few minutes...)\n")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        
        dtype = torch.float32,
        device_map  = "cpu"
    )

    if hasattr(model, "generation_config") and model.generation_config.max_length:
        model.generation_config.max_length = None

    llm_pipeline = pipeline(
        "text-generation",
        model     = model,
        tokenizer = tokenizer,
    )

    print("[llm_handler] Model loaded successfully.")
    return llm_pipeline


# Inference
def generate_answer(query: str, retrieved: list[dict], llm_pipeline) -> str:
    """
    Builds the prompt, runs inference with explicit generation params,
    and returns only the assistant's response text.
    """
    prompt = build_prompt(query, retrieved)

    output = llm_pipeline(
        prompt,
        max_new_tokens     = MAX_NEW_TOKENS,
        do_sample          = False,
        repetition_penalty = REPETITION_PENALTY,
    )

    return extract_response(output[0]["generated_text"])
