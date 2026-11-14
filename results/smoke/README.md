# Smoke-test artifacts (CPU sandbox, tiny scale; NOT results)

Produced by `scripts/smoke_test.sh` on a 2-core CPU sandbox (2026-11-14):
1.38M-param `tiny` model, 4k smoke tokenizer, 3.0M streamed tokens (EN/ES/FR), 300 steps.

- `train_log.csv`, `val_log.csv`, `train_samples.txt` – pretraining logs (stopped at 150, auto-resumed to 300)
- `loss_smoke_base.png`, `ppl_per_lang_smoke_base.png`, `bpb_smoke_base.png` – charts
- `eval_smoke_base.json`, `samples_smoke_base.md` – evaluation of the smoke base model
- `sft_log.csv`, `eval_smoke_sft.json`, `samples_smoke_sft.md` – 40-step SFT smoke
- `fertility_sandbox32k.{json,png}` – a 32k tokenizer trained on 30 MB/lang, vs GPT-2, on 200 held-out docs/lang
- `data_meta.json` – shard statistics
