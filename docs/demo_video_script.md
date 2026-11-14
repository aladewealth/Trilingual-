# Trilingua: demo video script (2:30 target, 3:00 max)

Format: screen recording with voice-over and the face cam in a corner (optional). Record at 1080p. Bracketed items are what is on screen. Numbers marked TODO come from the real run. Do not say any number on camera that is not in `results/`.

---

### 0:00–0:15 · Hook
[Gradio app, French selected, text streaming in]
**VO:** "This model writes in English, Spanish and French. I didn't fine-tune it from someone else's model. I built every piece from scratch: the tokenizer, the data pipeline, the transformer and the training loop. Then I trained it on free Kaggle GPUs."

### 0:15–0:40 · The problem: tokenizers are English-first
[`fertility.png` full screen]
**VO:** "Standard tokenizers like GPT-2's split Spanish and French into many more pieces than English. That makes them slower and more expensive, and a small model has to work harder. So I trained a shared 32k-token BPE on balanced English, Spanish and French text. It uses about TODO% fewer tokens for Spanish and TODO% fewer for French."

### 0:40–1:10 · How it's built
[README architecture diagram, then a scroll through `trilingua/model.py`]
**VO:** "It's a 70-million-parameter decoder-only transformer, written in plain PyTorch: rotary position embeddings, RMSNorm, SwiGLU and tied embeddings. The training data is FineWeb-Edu for English and FineWeb2-HQ for Spanish and French, mixed 50/25/25. Every document starts with a language tag, which is how you choose the output language."
[`02_pretrain.ipynb` running: two T4s, log lines scrolling]
**VO:** "Training runs on two T4 GPUs with DDP and mixed precision. Kaggle sessions stop after 12 hours, so the trainer checkpoints everything, down to the exact position in the data. I tested that resuming gives bit-for-bit the same model."

### 1:10–1:40 · Results
[`loss_base.png`, then `bpb_base.png`, then the table in the README]
**VO:** "This is the loss over TODO billion tokens and TODO GPU-hours. Per-language validation perplexity: English TODO, Spanish TODO, French TODO. Perplexity can't be compared across tokenizers, so for GPT-2 I compare bits per byte on the same held-out text. GPT-2 is almost twice our size. [state the TODO result honestly, including where GPT-2 wins, e.g. English]."
[language-ID numbers]
**VO:** "Does the French tag produce French? A fastText language detector says yes TODO% of the time."

### 1:40–2:20 · Live demo
[Gradio app on CPU]
1. English, *Continue text*: "Once upon a time, in a quiet harbour town," → Generate (streaming).
2. Switch to Español and the same mode, with the default example.
3. Français, *Write about a topic*: "l'histoire du chocolat" (SFT model).
4. Move the temperature slider from 0.3 to 1.1 and show the difference in one sentence.

**VO:** "This runs on a laptop CPU. Pick a language, a mode and your sampling settings. With a small instruction-tuning stage, it can also write about a topic you give it."

### 2:20–2:40 · Honest limits and close
[README Limitations, then the repo URL and HF model URL]
**VO:** "It's small, so it sounds fluent but often gets facts wrong. Treat it as a writing toy, not a source of truth. The code, the tests, the Kaggle notebooks and the weights are all open, so you can reproduce it on free GPUs. This is Trilingua."

---

**Shot checklist:** fertility chart · architecture diagram · model.py scroll · Kaggle T4×2 training log · loss curve · BPB chart · language-ID table · Gradio demo in all three languages · repo and HF links.
**Before recording:** fill in the TODOs from `results/eval_base.json`, `results/eval_sft.json`, `fertility.json` and the train logs. Rehearse at about 150 words per minute: the voice-over is about 330 words (about 2:15), which leaves room for the demo pauses.
