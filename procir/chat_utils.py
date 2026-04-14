_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
_THINK_BLOCK = "<think>\n\n</think>\n\n"


def patch_think_tokens(text_str: str) -> str:
    """Ensure every assistant turn starts with <think>...</think>."""
    parts = text_str.split(_ASSISTANT_PREFIX)
    if len(parts) <= 1:
        return text_str
    result = parts[0]
    for part in parts[1:]:
        result += _ASSISTANT_PREFIX
        if not part.startswith("<think>"):
            result += _THINK_BLOCK
        result += part
    return result
