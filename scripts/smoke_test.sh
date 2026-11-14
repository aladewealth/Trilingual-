#!/usr/bin/env bash
# End-to-end CPU smoke test at tiny scale: stream data -> tokenizer -> shards -> train -> resume -> eval -> generate.
# Numbers from this script are a SMOKE TEST ONLY (1.4M params, ~1.2M tokens), not results.
# Each step is restartable; the sandbox ran it in <2-minute chunks.
set -euo pipefail
cd "$(dirname "$0")/.."
S=${SMOKE_DIR:-/tmp/trilingua_smoke}
D=$S/data
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

# 1) data: 30 MB/lang sample, 4k smoke tokenizer (+ a 32k tokenizer for fertility), 3M-token shards
python -m trilingua.prepare_data sample --out_dir $D --mb_per_lang 30 --val_every 20
python -m trilingua.prepare_data tokenizer --out_dir $D --vocab_size 4096 --tokenizer $D/tokenizer.json
python -m trilingua.prepare_data tokenizer --out_dir $D --vocab_size 32768 --tokenizer $D/tokenizer_32k.json
python -m trilingua.prepare_data shards --out_dir $D --total_tokens 3e6 --val_tokens 6e4 --val_every 20 \
    --shard_tokens 1000000 --save_val_docs 200
python -m trilingua.prepare_data fertility --tokenizer $D/tokenizer_32k.json --data_dir $D \
    --json_out results/smoke/fertility_sandbox32k.json --png_out results/smoke/fertility_sandbox32k.png

# 2) train 300 steps in two sessions (stop at 150, auto-resume to 300)
python -m trilingua.train --config_file configs/smoke_tiny.json --data_dir $D --out_dir $S/out_a --stop_step 150
python -m trilingua.train --config_file configs/smoke_tiny.json --data_dir $D --out_dir $S/out_a

# 3) bitwise resume check: 40 steps straight vs 20 + resume 20
python -m trilingua.train --config_file configs/smoke_tiny.json --data_dir $D --out_dir $S/res_cont --max_steps 40 --eval_interval 1000 --sample_tokens 0
python -m trilingua.train --config_file configs/smoke_tiny.json --data_dir $D --out_dir $S/res_split --max_steps 40 --eval_interval 1000 --sample_tokens 0 --stop_step 20
python -m trilingua.train --config_file configs/smoke_tiny.json --data_dir $D --out_dir $S/res_split --max_steps 40 --eval_interval 1000 --sample_tokens 0
python scripts/compare_ckpts.py $S/res_cont/ckpt_latest.pt $S/res_split/ckpt_latest.pt

# 4) eval + SFT + export + generate
python -m trilingua.evaluate --model $S/out_a/ckpt_latest.pt --data_dir $D --out_dir results/smoke --langid_n 20 \
    --max_new 48 --n_docs 100 --train_dir $S/out_a --tag smoke_base
python -m trilingua.sft build --data_dir $S/sft_data --cont_per_lang 100
python -m trilingua.sft train --data_dir $S/sft_data --tokenizer $D/tokenizer.json --init_ckpt $S/out_a/ckpt_latest.pt \
    --out_dir $S/sft_out --max_steps 40 --micro_batch 8 --grad_accum 2 --lr 1e-3 --warmup_steps 5 --eval_interval 20
python -m trilingua.export --ckpt $S/out_a/ckpt_latest.pt --tokenizer $D/tokenizer.json --out_dir $S/export
python -m trilingua.export --ckpt $S/sft_out/sft_final.pt --tokenizer $D/tokenizer.json --out_dir $S/export/sft
python generate.py --model $S/export --lang fr --prompt "Il était une fois" --max_new_tokens 30 --seed 0
python generate.py --model $S/export --subfolder sft --lang es --mode write --prompt "el océano" --max_new_tokens 30 --seed 0
echo "smoke test complete"
