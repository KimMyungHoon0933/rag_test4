"""Ollama generation settings shared by Streamlit and LangServe."""
DEFAULT_MODEL = "Llama3_ko_8b_Q8:latest"
DEFAULT_DOCUMENT_MODEL = DEFAULT_MODEL
CUSTOM_Q8_NAME = DEFAULT_MODEL.partition(":")[0].casefold()
LEGACY_ALPACA_NAME = "llama_unslot_q82"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{{ .Prompt }}

### Input:


### Response:
{{ .Response }}"""


def uses_short_prompt(model: str) -> bool:
    return model.partition(":")[0].casefold() in {CUSTOM_Q8_NAME, LEGACY_ALPACA_NAME}


def make_llm(model: str, base_url: str, *, document_reply: bool = False):
    from langchain_ollama import ChatOllama, OllamaLLM

    options = {
        "model": model,
        "base_url": base_url,
        "temperature": 0,
        "num_predict": 120,
        "repeat_penalty": 1.15,
        "client_kwargs": {"timeout": 180},
    }
    if document_reply:
        from app.document_answers import DOCUMENT_REPLY_SCHEMA

        options.update(num_predict=256, format=DOCUMENT_REPLY_SCHEMA)
    model_name = model.partition(":")[0].casefold()
    if model_name in {CUSTOM_Q8_NAME, LEGACY_ALPACA_NAME}:
        reply_format = options.pop("format", None)
        llm = OllamaLLM(**options, num_ctx=4096)
        if model_name == LEGACY_ALPACA_NAME:
            return llm.bind(
                template=ALPACA_TEMPLATE, format=reply_format,
                stop=["### Instruction:", "### Input:", "### Response:", "### Response is:", "[INST]"],
            )
        return llm.bind(format=reply_format)
    return ChatOllama(**options, num_ctx=8192)
