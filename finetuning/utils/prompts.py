SYSTEM_PROMPT = (
"You are a careful assistant for extractive question answering. "
"Answer strictly using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
)


# Render a single-turn chat sample to a model-specific template


def render_chat(chat_template: str, context: str, question: str, answer: str):
    user = (
    "Answer the question strictly based on the context.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {question}\n"
    "Answer:"
    )
    if chat_template == "llama":
        # Llama 3 Instruct chat format
        return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{answer}<|eot_id|>"
        )
    elif chat_template == "qwen":
    # Qwen2.5 chat format (bos is added by tokenizer usually)
        return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>\n"
        )
    else:
        # Generic fallback
        return f"[SYSTEM]\n{SYSTEM_PROMPT}\n[USER]\n{user}\n[ASSISTANT]\n{answer}\n"