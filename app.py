"""Gradio demo for Trilingua (runs on CPU).

  python app.py                                   # loads aladewealth/trilingua from the HF Hub
  TRILINGUA_MODEL=export python app.py            # or a local exported folder
  TRILINGUA_SFT_SUBFOLDER=sft python app.py       # SFT weights in a subfolder of the same repo
"""
import os

import gradio as gr
import torch

from trilingua.inference import build_prompt, load_bundle, stream_text

MODEL = os.environ.get("TRILINGUA_MODEL", "aladewealth/trilingua")
SFT_SUB = os.environ.get("TRILINGUA_SFT_SUBFOLDER", "sft")
torch.set_grad_enabled(False)
torch.set_num_threads(max(1, os.cpu_count() or 1))

BUNDLES = {"base": load_bundle(MODEL)}
try:
    BUNDLES["sft"] = load_bundle(MODEL, subfolder=SFT_SUB)
except Exception as e:  # SFT weights are optional
    print(f"no SFT weights ({e.__class__.__name__}); 'Write about a topic' uses the base model")

LANGS = {"English": "en", "Español": "es", "Français": "fr"}
EXAMPLES = {
    "en": ("Once upon a time, in a quiet harbour town,", "the history of chocolate"),
    "es": ("Había una vez, en un pueblo junto al mar,", "la historia del chocolate"),
    "fr": ("Il était une fois, dans un petit port tranquille,", "l'histoire du chocolat"),
}


def run(lang_name, mode, text, max_new, temperature, top_k, top_p, rep):
    lang = LANGS[lang_name]
    if not text.strip():
        text = EXAMPLES[lang][0 if mode == "Continue text" else 1]
    if mode == "Continue text":
        b, m, prefix = BUNDLES["base"], "continue", text
    else:
        b = BUNDLES.get("sft", BUNDLES["base"])
        m, prefix = "write", ""
    ids = build_prompt(b, lang, m, text)
    for out in stream_text(b, ids, int(max_new), float(temperature), int(top_k), float(top_p), float(rep)):
        yield (prefix + out) if prefix else out


with gr.Blocks(title="Trilingua") as demo:
    gr.Markdown("# Trilingua\nA small English / Spanish / French language model trained from scratch "
                "(tokenizer, data pipeline, model and training loop). Runs on CPU.")
    with gr.Row():
        with gr.Column(scale=1):
            lang = gr.Radio(list(LANGS), value="English", label="Language")
            mode = gr.Radio(["Continue text", "Write about a topic"], value="Continue text", label="Mode")
            max_new = gr.Slider(16, 400, value=160, step=8, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature (0 = greedy)")
            top_k = gr.Slider(0, 200, value=50, step=1, label="Top-k (0 = off)")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            rep = gr.Slider(1.0, 2.0, value=1.1, step=0.01, label="Repetition penalty")
        with gr.Column(scale=2):
            text = gr.Textbox(lines=4, label="Text to continue / topic",
                              placeholder="Leave empty for an example in the selected language")
            go = gr.Button("Generate", variant="primary")
            out = gr.Textbox(lines=14, label="Output")
    go.click(run, [lang, mode, text, max_new, temperature, top_k, top_p, rep], out)
    gr.Markdown(f"Model: `{MODEL}`. " + ("SFT weights loaded." if "sft" in BUNDLES else "Base model only."))

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
