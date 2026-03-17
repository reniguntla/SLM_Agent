import threading
import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Tuple

import streamlit as st

try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TextIteratorStreamer,
    )
except Exception:  # transformers may not be installed in all environments
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None


MAX_TURNS = 10
MAX_CONTEXT_CHARS = 6000


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str


SUPPORTED_MODELS: Dict[str, ModelSpec] = {
    "Phi-3": ModelSpec("Phi-3", "microsoft/Phi-3-mini-4k-instruct"),
    "Gemma": ModelSpec("Gemma", "google/gemma-2b-it"),
    "Mistral": ModelSpec("Mistral", "mistralai/Mistral-7B-Instruct-v0.2"),
}


@st.cache_resource(show_spinner=True)
def load_model(model_id: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        return None, None, "Transformers is not installed. Returning a local fallback response."

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch and torch.cuda.is_available() else None,
            device_map="auto" if torch and torch.cuda.is_available() else None,
        )
        return tokenizer, model, None
    except Exception as exc:
        return None, None, f"Model unavailable ({model_id}): {exc}"


def _format_history(history: List[Tuple[str, str]]) -> str:
    lines: List[str] = []
    for user_q, ai_a in history:
        lines.append(f"User: {user_q}")
        lines.append(f"Assistant: {ai_a}")
    return "\n".join(lines)


def build_prompt(history: List[Tuple[str, str]], question: str) -> str:
    trimmed = history.copy()
    while len(_format_history(trimmed)) > MAX_CONTEXT_CHARS and trimmed:
        trimmed.pop(0)

    context = _format_history(trimmed)
    if context:
        return (
            "You are a helpful assistant. Use the chat history for context when relevant.\n\n"
            f"Chat history:\n{context}\n\n"
            f"User: {question}\nAssistant:"
        )
    return f"You are a helpful assistant.\nUser: {question}\nAssistant:"


def _fallback_stream(history: List[Tuple[str, str]], question: str) -> Generator[str, None, None]:
    context_hint = ""
    if history:
        context_hint = f" I considered the last {len(history)} interactions."
    text = (
        "[Fallback response] I could not load the selected SLM in this environment."
        f" Your question was: '{question}'.{context_hint}"
    )
    for token in text.split(" "):
        yield token + " "
        time.sleep(0.02)


def stream_response(model_name: str, history: List[Tuple[str, str]], question: str) -> Generator[str, None, None]:
    model_spec = SUPPORTED_MODELS[model_name]
    tokenizer, model, err = load_model(model_spec.model_id)
    if err or tokenizer is None or model is None or TextIteratorStreamer is None:
        yield from _fallback_stream(history, question)
        return

    prompt = build_prompt(history, question)
    inputs = tokenizer(prompt, return_tensors="pt")

    if torch and torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
    }

    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text in streamer:
        yield text


def trim_history(history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if len(history) <= MAX_TURNS:
        return history
    return history[-MAX_TURNS:]


def main():
    st.set_page_config(page_title="SLM Chat", page_icon="💬", layout="centered")
    st.title("💬 SLM Conversational Assistant")

    if "history" not in st.session_state:
        st.session_state.history = []

    model_name = st.selectbox("SLM Selection", options=list(SUPPORTED_MODELS.keys()), index=0)
    st.markdown(f"**Model Status: Currently Using {model_name}**")

    st.divider()
    st.subheader("Conversation")
    if not st.session_state.history:
        st.info("No conversation yet. Ask a question to get started.")
    else:
        for user_q, ai_a in st.session_state.history:
            with st.chat_message("user"):
                st.write(user_q)
            with st.chat_message("assistant"):
                st.write(ai_a)

    st.divider()
    question = st.text_input("Question Input", placeholder="Type your question here")
    col1, col2 = st.columns([1, 1])

    with col1:
        submit_clicked = st.button("Submit", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear Conversation", use_container_width=True)

    if clear_clicked:
        st.session_state.history = []
        st.rerun()

    if submit_clicked and question.strip():
        with st.chat_message("user"):
            st.write(question)

        assistant_box = st.chat_message("assistant")
        placeholder = assistant_box.empty()

        built = ""
        for chunk in stream_response(model_name, st.session_state.history, question):
            built += chunk
            placeholder.write(built)

        st.session_state.history.append((question, built.strip()))
        st.session_state.history = trim_history(st.session_state.history)


if __name__ == "__main__":
    main()
