# Trilingua

**A ~70M-parameter English / Spanish / French language model trained from scratch on free Kaggle GPUs.**
We wrote and trained every part ourselves: the tokenizer, the streaming data pipeline, the transformer, the DDP training loop, the evaluation and the demo.

> Arbiter Hacks V1, General Track. "Trilingua" is a working name.
>
> **Status:** the code is complete and tested on CPU at tiny scale (see [Smoke test](#smoke-test-cpu-sandbox-not-a-result)). The full pretraining run on Kaggle has **not** happened yet. Every number marked **TODO** comes from that run and will be filled in from its logs. Every number in this README that is not marked TODO came from an actual run.

---

## What it does

- Writes and continues text in **English, Spanish or French**. You pick the language with a tag token (`<|en|>`, `<|es|>`, `<|fr|>`) that also starts every pretraining document.
- An optional small **SFT stage** turns it into a writing assistant: "write a paragraph about X" or "continue this story" in any of the three languages.
- A **Gradio demo** runs on CPU. It has a language selector, two modes (continue text or write about a topic) and sampling sliders. There is also a streaming `generate.py` CLI.

## Architecture

```
 token ids ──► Embedding (32768 × 640) ─────────────────────────────┐ (weights tied with LM head)
                                                                    │
            ┌──────────────── × 10 blocks (pre-norm) ───────────────┤
            │  x ─► RMSNorm ─► Multi-head attention ─(+)─► x'       │
            │        │   q,k ◄─ RoPE (θ=10k)    ▲                   │
            │        │   SDPA causal (10 heads × 64)                │
            │  x'─► RMSNorm ─► SwiGLU MLP (640→1728→640) ─(+)─► x'' │
            └───────────────────────────────────────────────────────┤
                                                                    │
                   final RMSNorm ─► LM head (= Embeddingᵀ) ─► logits over 32768 tokens
```

| | value |
|---|---|
| layers / d_model / heads / head_dim | 10 / 640 / 10 / 64 |
| MLP | SwiGLU, hidden 1728 (8/3·d rounded up to a multiple of 64) |
| positions | RoPE, θ = 10 000, context 1024 |
| norm | RMSNorm (pre-norm), fp32 internally |
| attention | `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`, GQA-capable (`n_kv_head`) |
| embeddings | tied input/output, vocab 32 768 |
| parameters | **70.55M total**, 49.58M non-embedding (counted, and checked against the analytic formula in the tests) |
| training FLOPs/token | 0.502 GFLOP (6·N_matmul + 12·L·d·T) |

### Candidate configs (32k vocab, context 1024)

Throughput is a **FLOP-based estimate** for 2× T4 (65 TFLOPS fp16 peak each, 95% DDP scaling). It is not a measurement. `scripts/estimate_throughput.py` produced this table; `--benchmark_steps` measures the real numbers on Kaggle.

| config | d_model | layers | heads | params | non-emb | GFLOP/tok | 20·N tokens | est. tok/s @20% MFU | @30% | est. h for 1.25B @20% | @30% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 512 | 8 | 8 | 42.5M | 25.7M | 0.305 | 0.85B | 81k | 121k | 4.3 | 2.9 |
| small12 | 512 | 12 | 8 | 55.3M | 38.5M | 0.407 | 1.11B | 61k | 91k | 5.7 | 3.8 |
| **base (default)** | 640 | 10 | 10 | **70.5M** | 49.6M | 0.502 | 1.41B | 49k | 74k | 7.1 | 4.7 |
| base12 | 640 | 12 | 10 | 80.5M | 59.5M | 0.577 | 1.61B | 43k | 64k | 8.1 | 5.4 |
| wide | 768 | 8 | 12 | 81.8M | 56.6M | 0.566 | 1.64B | 44k | 65k | 8.0 | 5.3 |

`base` matches our token budget: 1.25B tokens is about 18 tokens per parameter, close to Chinchilla. The `wide` config would sit at about 1.64B tokens for 20× and would not fit the GPU budget. Measured T4 throughput: **TODO** (`02_pretrain.ipynb`, `BENCHMARK=True`).

## Design choices and why

| choice | why |
|---|---|
| **Our own 32k byte-level BPE** shared by all three languages, NFC-normalised | GPT-2's tokenizer spends about 50% more tokens on Spanish and French (see fertility below). A shared vocabulary lets subwords like *-ción / -tion* be reused across languages. Byte-level means no unknown tokens. NFC means `é` typed as one code point or two gives the same tokens. |
| **Language tag on every document** (`<|lang|> text <|endoftext|>`) | One token makes the language controllable at inference, and we can measure that control (language-ID accuracy of tagged generations). |
| **50/25/25 mix sampled at load time** | The loader picks each sample's language with a seeded RNG, so the mix is a training-time knob and per-language shards stay separate for per-language validation. |
| **FineWeb2-HQ for ES/FR** | It is the FineWeb-Edu-style, quality-filtered top slice of FineWeb-2 (ODC-By). For a small model, cleaner data is worth more than more data. `--source fineweb2` switches to plain FineWeb-2. |
| **RoPE, RMSNorm, SwiGLU, pre-norm, tied embeddings** | This is the standard modern recipe (LLaMA-style). It is stable in fp16 and gives the best quality per parameter. Tying saves 21M parameters (30% of the model) at a 32k vocab. |
| **SDPA, no flash-attn dependency** | T4 (sm75) cannot run FlashAttention-2. PyTorch SDPA picks the memory-efficient kernel instead. |
| **fp16 + GradScaler** | T4 has no bf16. RMSNorm, RoPE and the loss run in fp32 for stability. `--dtype auto` picks bf16 on GPUs that support it. |
| **Stateless-resumable data loader** | All ranks hold the same loader state (RNG plus a per-language permutation cursor), draw the global micro-batch and keep their own slice. A checkpoint stores that small state, so resume is **bitwise exact** (tested). |
| **Time-budgeted sessions** | `--max_hours` checkpoints and exits cleanly before Kaggle's 12h limit. `--resume auto --resume_dirs /kaggle/input` picks up the newest checkpoint from a previous session's output and carries its logs forward. |
| **Bits-per-byte as the cross-model metric** | Per-token perplexity cannot be compared across tokenizers. BPB on identical raw text can, so the GPT-2 comparison uses BPB. Per-token PPL is still reported for our own model. |

## Data and licenses

Configs and licenses were checked against the Hugging Face Hub API on 2026-11-14 (`/api/datasets/<id>`, `cardData.configs` and `cardData.license`):

| use | dataset | config / split | license |
|---|---|---|---|
| English pretraining | `HuggingFaceFW/fineweb-edu` | `sample-10BT` / `train` | ODC-By 1.0 |
| Spanish pretraining | `epfml/FineWeb2-HQ` | `spa_Latn` / `train` | ODC-By 1.0 |
| French pretraining | `epfml/FineWeb2-HQ` | `fra_Latn` / `train` | ODC-By 1.0 |
| alternative ES/FR | `HuggingFaceFW/fineweb-2` | `spa_Latn`, `fra_Latn` / `train` (+`test`) | ODC-By 1.0 |
| SFT | `CohereLabs/aya_dataset` (formerly CohereForAI) | `default` / `train`, filtered to `eng` 3,944, `spa` 3,854, `fra` 1,422 rows | Apache-2.0 |
| comparison only | `gpt2` (OpenAI) | n/a | MIT |

All the FineWeb sets are built from Common Crawl, so Common Crawl's terms of use apply as well. FineWeb2-HQ rows include a large `embeddings` column, so we stream with `columns=["text"]` and download only the text. The SFT set also includes self-supervised "continue this text" examples cut from pretraining-source documents (ODC-By), with instructions written in each document's language.

**Pipeline** (`trilingua/prepare_data.py`, run by `01_tokenizer_and_data.ipynb` on a CPU notebook, so it uses no GPU quota):

1. `sample`: streams a balanced raw sample (default 150 MB per language) for the tokenizer.
2. `tokenizer`: trains the 32 768-token BPE. The 9 special tokens get fixed ids 0–8: `<|endoftext|> <|pad|> <|en|> <|es|> <|fr|> <|system|> <|user|> <|assistant|> <|end|>`.
3. `shards`: tokenizes the streams into `train_{lang}_NNNN.bin` and `val_{lang}_NNNN.bin` (uint16, 256×int32 header, 100M tokens per shard) up to a 1.3B-token budget split 50/25/25, plus 2M val tokens per language.
4. `fertility`: tokens per word against GPT-2 on held-out documents.

Document *i* of each stream goes to validation if *i* mod 100 = 0, until the validation quota is met. Validation documents never enter the training shards or the tokenizer sample.

## Reproduce on Kaggle

Steps 1–3 below have not been run on Kaggle yet.

1. **Code.** Push this repo to GitHub (set `REPO_URL` in the notebooks), or upload the folder as a Kaggle Dataset called `trilingua-code` and attach it.
2. **`notebooks/01_tokenizer_and_data.ipynb`**: Accelerator None, Internet On, *Save & Run All*. Its output (`/kaggle/working/data`, about 2.6 GB of shards plus `tokenizer.json`, `meta.json`, `fertility.png`) is the input to the other notebooks. If the session runs out of time, set `LANGS` to the unfinished languages and run it again; finished languages are skipped.
3. **`notebooks/02_pretrain.ipynb`**: Accelerator **GPU T4 ×2**. Attach the 01 output. Optionally run `BENCHMARK=True` once to measure tok/s per config. Then *Save & Run All*. It runs
   `torchrun --standalone --nproc_per_node=2 -m trilingua.train --config_file configs/pretrain_base.json --resume auto --resume_dirs /kaggle/input --max_hours 11`.
   **To continue in a new session**, add the previous version's output of this notebook as an input and run it again. It resumes from the newest `ckpt_latest.pt` (model, optimizer, GradScaler, step, data position, logs).
   Optional secret: `WANDB_API_KEY`. Without it, logs go to CSV only.
4. **`notebooks/03_sft.ipynb`** (one T4 is enough): builds the SFT mix and fine-tunes for 2 epochs. Output: `sft_out/sft_final.pt`.
5. **`notebooks/04_eval_and_export.ipynb`**: per-language PPL and BPB, the GPT-2 BPB comparison, language-ID of generations, charts, safetensors export. With the `HF_TOKEN` secret it pushes base weights to `aladewealth/trilingua` (placeholder) and SFT weights to `sft/`.
6. **Demo:** `pip install -r requirements.txt && python app.py` (CPU). It loads from the HF repo, or from `TRILINGUA_MODEL=/path/to/export`.

Local / CLI equivalents:

```bash
python -m trilingua.prepare_data sample|tokenizer|shards|fertility --out_dir data
torchrun --standalone --nproc_per_node=2 -m trilingua.train --config_file configs/pretrain_base.json
python -m trilingua.sft build --data_dir sft_data && python -m trilingua.sft train --init_ckpt out/ckpt_latest.pt --tokenizer data/tokenizer.json
python -m trilingua.evaluate --model out/ckpt_latest.pt --data_dir data --compare gpt2 --train_dir out
python -m trilingua.export --ckpt out/ckpt_latest.pt --tokenizer data/tokenizer.json --out_dir export [--push aladewealth/trilingua]
python generate.py --model export --lang es --mode write --prompt "los volcanes de México" --temperature 0.8 --top_k 50 --top_p 0.95 --repetition_penalty 1.1
python -m pytest -q
bash scripts/smoke_test.sh        # the full tiny-scale CPU pipeline
```

## Compute budget

Kaggle gives about 30 GPU-hours a week, shared with another project. The plan uses about 8–10.

| stage | hardware | estimate | source |
|---|---|---|---|
| data + tokenizer | Kaggle CPU notebook | 0 GPU-h (wall time TODO) | — |
| throughput benchmark | 2× T4 | ~0.3 h | — |
| pretraining, 1.25B tokens, 2385 steps × 524 288 tokens | 2× T4 | **7.1 h at 20% MFU** (4.7 h at 30%), plus about 5% for eval and checkpoints | FLOP estimate |
| SFT (~13k examples × 2 epochs, ~830 steps) | 1× T4 | ~0.3 h | rough estimate |
| eval + export | 1× T4 | ~0.3 h | rough estimate |

**Decision rule after the benchmark:** if measured 2×T4 throughput is under about 40k tok/s (2385 steps would take over 8.7 h), set `--total_tokens 1.0e9` or switch to `small12` before starting. The cosine schedule is fixed by `total_tokens`, so do not change it mid-run. Training spans sessions through `--max_hours 11` plus auto-resume.

## Results

All TODO until the Kaggle run. Values will come from `results/eval_base.json`, `results/eval_sft.json` and the training CSVs.

| metric | EN | ES | FR |
|---|---|---|---|
| tokenizer fertility, Trilingua 32k (tokens/word) | TODO | TODO | TODO |
| tokenizer fertility, GPT-2 | TODO | TODO | TODO |
| val loss (nats/token) | TODO | TODO | TODO |
| val perplexity | TODO | TODO | TODO |
| bits per byte, Trilingua (raw val docs) | TODO | TODO | TODO |
| bits per byte, GPT-2 small 124M (same docs) | TODO | TODO | TODO |
| language-ID accuracy of `<|lang|>` generations (base) | TODO | TODO | TODO |
| language-ID accuracy (SFT, chat prompts) | TODO | TODO | TODO |

- Training throughput (measured): TODO tok/s, MFU TODO, GPU-hours used TODO
- Charts: `results/loss_base.png`, `results/ppl_per_lang_base.png`, `results/bpb_base.png`, `fertility.png` (TODO)
- Samples: `results/samples_base.md`, `results/samples_sft.md` (TODO)

**Early tokenizer number (sandbox, not final).** We trained a 32k tokenizer on a smaller 30 MB-per-language sample in the dev sandbox and measured it on 200 held-out documents per language: **Trilingua EN 1.45 / ES 1.42 / FR 1.51 tokens per word, against GPT-2 at 1.35 / 2.11 / 2.14.** That is about 33% fewer tokens for Spanish and 30% fewer for French, at a 7% cost on English. The final tokenizer is trained on 150 MB per language in notebook 01 and its numbers replace these (`results/smoke/fertility_sandbox32k.*`).

## Smoke test (CPU sandbox, not a result)

These numbers come from `scripts/smoke_test.sh` on a 2-core CPU sandbox. They show that the pipeline works end to end. They say nothing about model quality.

- **Data:** real streamed text. 3.0M training tokens (EN 1.50M, ES 0.75M, FR 0.75M) and about 60k val tokens per language, encoded with a 4k smoke tokenizer.
- **Model `tiny`:** d=128, 4 layers, 4 heads, context 128, **1.38M params**. Batch 4096 tokens, 300 steps, lr 3e-3, fp32.
- **Loss:** train **7.77 at step 10 → 5.96 at step 300** (ln 4096 = 8.32 at init). Mean val loss **6.99 at step 50 → 5.97 at step 300** (EN 6.87→5.96, ES 7.03→5.90, FR 7.07→6.04).
- **Resume:** the run stopped at step 150 and auto-resumed to 300, and the logs continue across the stop. Bitwise check: 40 steps straight versus 20 steps, checkpoint, 20 resumed gave **identical model, AdamW state and loader state (max |Δw| = 0)**.
- **DDP:** `torchrun --nproc_per_node=2` (gloo, CPU) with micro-batch 8 per rank gave **exactly the same losses** at steps 10 and 20 (7.77024, 7.18101) as one process with micro-batch 16. So rank partitioning and gradient averaging match single-GPU training.
- **SFT smoke:** 40 steps on aya plus continuation data, val loss 6.144 → 5.895.
- **Export / demo:** the safetensors export reproduces the checkpoint's logits exactly. `generate.py` streams from a checkpoint, an export folder and the `sft/` subfolder. `app.py` serves HTTP 200 on CPU.
- **Samples are gibberish, as expected at this size.** Language-ID of tagged generations is 35% / 5% / 20% for EN / ES / FR, because 1.2M training tokens is too few to learn language control. The same fastText classifier scores 100% on real val text in all three languages, so this measures the model, not the classifier.

## Tests

`python -m pytest -q`: **46 passed** in this sandbox. That count includes 6 round-trips of a trained 32k tokenizer, enabled with `TRILINGUA_TOKENIZER=...`; without that variable the result is 40 passed and 6 skipped. The tests cover:

- **model:** shapes, analytic parameter count for every preset, tied weights, causal masking on logits and on gradients, KV cache equal to a full forward pass for MHA, GQA and MQA (including chunked prefill), scaled init
- **RoPE:** norm preservation, identity at position 0, relative-position property, dtype
- **tokenizer:** fixed special ids, round-trip on EN/ES/FR with `é ñ ç ¿ ¡ « » œ`, NFD→NFC, bytes-per-token accounting, chat loss masks
- **data loader:** shard I/O, mixture proportions, one epoch with no repeats, exact state resume, DDP rank partitioning
- **checkpointing:** bitwise resume, cross-session log carry-over, loss decreases
- **sampling:** top-k, top-p, repetition penalty, stop tokens, generation past the context length
- **language-ID heuristic**

## Limitations

- The model is small (70M) and trained on about 1.25B tokens. Expect fluent-looking but frequently wrong or incoherent text, especially beyond a paragraph. It is a writing toy and a learning artifact, not a knowledge source.
- The data is web text, filtered for educational quality, but it can still contain biases, errors and offensive content. Nothing in the pipeline adds safety tuning.
- The SFT data is small (about 9k human-written aya rows plus about 4.5k self-supervised continuations). French has the fewest aya rows (1,422), so its instruction following will likely be weakest.
- Context is 1024 tokens, and the model has seen no code or math data.
- The language-ID metric uses fastText lid.176 (a heuristic fallback exists) and only shows which language the output is in, not whether it is good.
- Throughput figures are estimates until the Kaggle benchmark runs.

## Repository layout

```
trilingua/            core package
  config.py           ModelConfig, presets, parameter/FLOP accounting
  model.py            transformer (RMSNorm, RoPE, SwiGLU, SDPA, KV cache)
  tokenizer.py        BPE training + TrilinguaTokenizer (lang tags, chat format, token bytes)
  prepare_data.py     HF streaming -> sample -> tokenizer -> shards -> fertility
  data.py             uint16 shards, resumable DDP mixture loader, val sets
  train.py            pretraining (DDP, fp16/GradScaler, compile, resume, time budget, CSV/wandb)
  sft.py              SFT data build + fine-tune with masked loss
  evaluate.py         PPL / BPB / GPT-2 compare / language-ID / samples / charts
  sampling.py         temperature, top-k, top-p, repetition penalty, streaming
  inference.py        load from dir / ckpt / HF Hub; prompt building
  export.py           safetensors + model card + HF push
  langid.py, plots.py, utils.py
generate.py           CLI
app.py                Gradio demo
configs/              pretrain_base.json, smoke_tiny.json, sft.json
notebooks/            01–04 Kaggle notebooks (generated by scripts/make_notebooks.py)
scripts/              smoke_test.sh, estimate_throughput.py, compare_ckpts.py, make_notebooks.py
tests/                pytest suite
results/smoke/        smoke-test logs, charts, samples (CPU, tiny scale)
docs/                 demo video script
```

---

## Hackathon write-up draft (Devpost)

TODOs will be replaced with real numbers after the Kaggle run.

**Inspiration.** Most "build an LLM" projects either fine-tune someone else's model or train an English-only toy. We wanted to know what one person can build from nothing on free GPUs if they own every layer. English, Spanish and French cover about a billion speakers, and they show a real flaw in English-first tooling: GPT-2's tokenizer spends about 50% more tokens on Spanish and French text.

**What it does.** Trilingua is a 70M-parameter language model that writes and continues text in English, Spanish and French. A single language-tag token chooses the output language. A short instruction-tuning stage lets it answer prompts like "write a paragraph about volcanoes" or "continue this story" in all three languages. A Gradio demo running on CPU streams the output and has a language selector and sampling controls.

**How we built it.**
- A 32k byte-level BPE trained on a balanced EN/ES/FR sample.
- A streaming pipeline from FineWeb-Edu and FineWeb2-HQ into uint16 shards, run on a free CPU notebook.
- A from-scratch PyTorch transformer (RoPE, RMSNorm, SwiGLU, SDPA, tied embeddings, KV cache).
- A DDP training loop for 2× T4 with fp16 + GradScaler and torch.compile, plus checkpoints that resume bitwise-exactly across 12-hour Kaggle sessions.
- Evaluation by per-language perplexity, bits-per-byte against GPT-2 on identical text, and fastText language-ID of generations.
- Pretraining used TODO GPU-hours for 1.25B tokens.

**Challenges.**
- Fitting Chinchilla-scale training into about 8–10 shared GPU-hours on T4s, which have no bf16 and no FlashAttention.
- Making training survive Kaggle's session limits without losing data order or optimizer state.
- Comparing fairly against a model with a different tokenizer, which is why we use bits per byte instead of perplexity.
- FineWeb2-HQ ships a heavy embedding column, so we stream only the text column.

**Accomplishments.**
- A tokenizer that cuts Spanish/French token counts by about 30% against GPT-2 (TODO final numbers).
- Bitwise-verified resume.
- DDP verified to match single-process training loss for loss.
- A 46-test suite.
- TODO: headline BPB and language-control results.

**What we learned.** TODO after training. Expected themes: how much tokenizer fertility matters for non-English quality at small scale, how data quality compares with quantity at 70M parameters, and fp16 stability details.

**What's next.** Temperature-based language sampling experiments, document-masked attention, a larger SFT set (especially French), more languages (Portuguese, Italian, Yoruba), and a quantized build for phones.

**Built with:** Python, PyTorch, Hugging Face `tokenizers` / `datasets` / Hub, Kaggle (2× T4), Gradio, fastText, matplotlib.
