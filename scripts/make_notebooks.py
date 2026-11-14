"""Generates the four Kaggle notebooks in notebooks/ (run: python scripts/make_notebooks.py)."""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks")

SETUP = r'''# --- Get the Trilingua code -------------------------------------------------------
# Option A (after the repo is public): clone it.
# Option B: upload the repo folder as a Kaggle Dataset named "trilingua-code" and attach it.
import os, sys, glob, shutil, subprocess
REPO_URL = "https://github.com/aladewealth/Trilingual-.git"
CODE = "/tmp/trilingua"
attached = glob.glob("/kaggle/input/**/trilingua/model.py", recursive=True)
if os.path.exists(CODE):
    pass
elif attached:
    shutil.copytree(os.path.dirname(os.path.dirname(attached[0])), CODE)
else:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, CODE], check=True)
os.chdir(CODE); sys.path.insert(0, CODE)
print("code at", CODE, os.listdir(CODE))'''

SECRETS = r'''# Optional secrets (Add-ons -> Secrets). Nothing breaks if they are missing.
def get_secret(name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(name)
    except Exception:
        return None
for k in ("WANDB_API_KEY", "HF_TOKEN"):
    v = get_secret(k)
    if v:
        os.environ[k] = v
        print(k, "loaded")'''

FIND_DATA = r'''# Locate the data produced by 01_tokenizer_and_data (attach that notebook's output as input)
metas = glob.glob("/kaggle/input/**/meta.json", recursive=True)
assert metas, "Attach the output of 01_tokenizer_and_data as an input dataset"
DATA = os.path.dirname(metas[0])
print("DATA =", DATA); print(open(os.path.join(DATA, "meta.json")).read())'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
    n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return n


NB1 = [
    ("md", "# 01 · Tokenizer and data (CPU notebook, no GPU quota)\n\n"
           "Streams EN (FineWeb-Edu `sample-10BT`) and ES/FR (FineWeb2-HQ `spa_Latn`/`fra_Latn`) from the Hugging Face Hub, "
           "trains the shared 32k byte-level BPE, measures fertility against GPT-2 and writes uint16 token shards.\n\n"
           "**Settings:** Accelerator = None, Internet = On. Run with *Save Version → Save & Run All*. "
           "The output (`/kaggle/working/data`) is the input of notebooks 02–04.\n\n"
           "If a session is too short for all three languages, set `LANGS` to a subset and run again: finished languages are skipped "
           "(`meta_<lang>.json`), and the outputs of several versions can be attached together."),
    ("code", SETUP),
    ("code", "!pip -q install -U 'tokenizers>=0.19' 'datasets>=3.0' 'huggingface_hub>=0.24' 2>&1 | tail -1\n"
             "!nproc; free -g; df -h /kaggle/working | tail -1"),
    ("code", "OUT = '/kaggle/working/data'\n"
             "LANGS = ['en', 'es', 'fr']\n"
             "TOTAL_TOKENS = 1.3e9        # ~1.25B used by pretraining + a small surplus\n"
             "MB_PER_LANG = 150           # balanced tokenizer sample\n"
             "os.makedirs(OUT, exist_ok=True)"),
    ("md", "## 1. Balanced raw-text sample for the tokenizer"),
    ("code", "!python -m trilingua.prepare_data sample --out_dir {OUT} --langs {' '.join(LANGS)} --mb_per_lang {MB_PER_LANG}"),
    ("md", "## 2. Train the 32k byte-level BPE (special tokens: `<|endoftext|> <|pad|> <|en|> <|es|> <|fr|> <|system|> <|user|> <|assistant|> <|end|>`)"),
    ("code", "if not os.path.exists(f'{OUT}/tokenizer.json'):\n"
             "    !python -m trilingua.prepare_data tokenizer --out_dir {OUT} --vocab_size 32768\n"
             "from trilingua.tokenizer import TrilinguaTokenizer\n"
             "tok = TrilinguaTokenizer.from_file(f'{OUT}/tokenizer.json')\n"
             "for s in ['¿Qué tal, señor Muñoz?', '« Ça va ? » dit le garçon à l’hôtel.', \"It's a test.\"]:\n"
             "    ids = tok.encode(s); assert tok.decode(ids) == s\n"
             "    print(len(ids), [tok.tok.id_to_token(i) for i in ids])"),
    ("md", "## 3. Tokenize to shards (three languages in parallel; ~50/25/25 mix)"),
    ("code", "import subprocess, time\n"
             "t0 = time.time()\n"
             "procs = [subprocess.Popen(['python', '-m', 'trilingua.prepare_data', 'shards', '--out_dir', OUT, '--langs', l,\n"
             "                           '--total_tokens', str(TOTAL_TOKENS), '--val_tokens', '2e6'],\n"
             "                          stdout=open(f'/kaggle/working/shards_{l}.log', 'w'), stderr=subprocess.STDOUT)\n"
             "         for l in LANGS]\n"
             "while any(p.poll() is None for p in procs):\n"
             "    time.sleep(300)\n"
             "    for l in LANGS:\n"
             "        lines = open(f'/kaggle/working/shards_{l}.log').read().strip().splitlines()\n"
             "        print(f'[{(time.time()-t0)/60:.0f} min]', lines[-1] if lines else '')\n"
             "print([p.returncode for p in procs])\n"
             "!python -m trilingua.prepare_data shards --out_dir {OUT}   # writes the combined meta.json"),
    ("md", "## 4. Fertility vs GPT-2 (tokens per word on held-out docs)"),
    ("code", "!python -m trilingua.prepare_data fertility --out_dir {OUT} --json_out {OUT}/fertility.json --png_out {OUT}/fertility.png\n"
             "from IPython.display import Image; Image(f'{OUT}/fertility.png')"),
    ("code", "# keep the output small: the raw tokenizer sample is no longer needed\n"
             "for f in glob.glob(f'{OUT}/tok_sample_*'):\n"
             "    os.remove(f)\n"
             "!du -sh {OUT}; ls -la {OUT}"),
]

NB2 = [
    ("md", "# 02 · Pretraining (GPU T4 ×2, resumable)\n\n"
           "**Settings:** Accelerator = GPU T4 ×2, Internet = On (only for pip/git), Persistence = Files.\n\n"
           "**Inputs:** the output of `01_tokenizer_and_data`. **To resume in a new session:** add *this notebook's previous version output* "
           "as an extra input. The trainer searches `/kaggle/input` for the newest `ckpt_latest.pt` and continues from it "
           "(model, optimizer, GradScaler, step, data-loader position, logs). It saves and exits cleanly before `MAX_HOURS`."),
    ("code", SETUP),
    ("code", SECRETS),
    ("code", FIND_DATA + "\n!nvidia-smi --query-gpu=name,memory.total --format=csv\nimport torch; print(torch.__version__, torch.cuda.device_count())"),
    ("code", "OUT = '/kaggle/working/out'\n"
             "MAX_HOURS = 11.0      # Kaggle sessions stop at 12 h\n"
             "NGPU = torch.cuda.device_count()\n"
             "BENCHMARK = False     # set True once to measure tok/s for candidate configs"),
    ("md", "## Optional: measure throughput (pick the config from measured tok/s)"),
    ("code", "if BENCHMARK:\n"
             "    for model, mb in [('small12', 8), ('base', 8), ('base', 16), ('base12', 8)]:\n"
             "        !torchrun --standalone --nproc_per_node={NGPU} -m trilingua.train --config_file configs/pretrain_base.json \\\n"
             "            --data_dir {DATA} --out_dir /kaggle/working/bench --model {model} --micro_batch {mb} --benchmark_steps 10 2>&1 | grep -E 'BENCHMARK|compile|Error'\n"
             "    !cat /kaggle/working/bench/benchmark.jsonl"),
    ("md", "## Train"),
    ("code", "!torchrun --standalone --nproc_per_node={NGPU} -m trilingua.train --config_file configs/pretrain_base.json \\\n"
             "    --data_dir {DATA} --out_dir {OUT} --resume auto --resume_dirs /kaggle/input --max_hours {MAX_HOURS}"),
    ("code", "from trilingua.plots import plot_losses\n"
             "plot_losses(f'{OUT}/train_log.csv', f'{OUT}/val_log.csv', f'{OUT}/loss.png')\n"
             "from IPython.display import Image; display(Image(f'{OUT}/loss.png'))\n"
             "!tail -n 12 {OUT}/samples.txt; ls -la {OUT}"),
]

NB3 = [
    ("md", "# 03 · Small SFT: trilingual writing assistant (GPU, 1×T4 is enough)\n\n"
           "**Inputs:** outputs of `01_tokenizer_and_data` and `02_pretrain`. Builds the SFT mix "
           "(CohereLabs/aya_dataset filtered to eng/spa/fra, Apache-2.0, plus self-supervised \"continue this text\" tasks in each language) "
           "and fine-tunes the pretrained model with loss on the replies only."),
    ("code", SETUP),
    ("code", FIND_DATA + "\n"
             "from trilingua.utils import find_latest_checkpoint\n"
             "BASE = find_latest_checkpoint(['/kaggle/input'])\n"
             "print('base checkpoint:', BASE)"),
    ("code", "!python -m trilingua.sft build --data_dir /kaggle/working/sft_data --cont_per_lang 1500\n"
             "!cat /kaggle/working/sft_data/sft_stats.json"),
    ("code", "!python -m trilingua.sft train --data_dir /kaggle/working/sft_data --tokenizer {DATA}/tokenizer.json \\\n"
             "    --init_ckpt {BASE} --out_dir /kaggle/working/sft_out --epochs 2 --micro_batch 8 --grad_accum 4 --lr 1e-4"),
    ("code", "from trilingua.inference import load_bundle, build_prompt, stream_text\n"
             "b = load_bundle('/kaggle/working/sft_out/sft_final.pt', tokenizer=f'{DATA}/tokenizer.json', device='cuda')\n"
             "for lang, topic in [('en', 'the ocean'), ('es', 'el océano'), ('fr', \"l'océan\")]:\n"
             "    out = list(stream_text(b, build_prompt(b, lang, 'write', topic), 120, seed=0))[-1]\n"
             "    print(f'[{lang}]', out, '\\n')"),
]

NB4 = [
    ("md", "# 04 · Evaluation, export and push to the Hugging Face Hub\n\n"
           "**Inputs:** outputs of 01, 02 and (optionally) 03. **Secret:** `HF_TOKEN` (write access). "
           "Computes per-language perplexity and bits-per-byte, compares against GPT-2 small on identical text (bits-per-byte, since "
           "per-token perplexity is not comparable across tokenizers), checks language-ID of generations, writes charts, exports "
           "safetensors and pushes base weights to the repo root and SFT weights to `sft/`."),
    ("code", SETUP),
    ("code", SECRETS),
    ("code", FIND_DATA + "\n"
             "from trilingua.utils import find_latest_checkpoint\n"
             "BASE = find_latest_checkpoint(['/kaggle/input'])\n"
             "SFT = find_latest_checkpoint(['/kaggle/input'], name='sft_final.pt')\n"
             "TRAIN_DIR = os.path.dirname(BASE)\n"
             "print('base', BASE, '\\nsft', SFT)\n"
             "!pip -q install fasttext-wheel 2>&1 | tail -1   # optional; falls back to a heuristic lang-id\n"
             "HF_REPO = 'aladewealth/trilingua'   # TODO: placeholder repo id\n"
             "RES = '/kaggle/working/results'"),
    ("code", "!python -m trilingua.evaluate --model {BASE} --data_dir {DATA} --out_dir {RES} --compare gpt2 \\\n"
             "    --train_dir {TRAIN_DIR} --tag base --langid_n 50 --device cuda"),
    ("code", "if SFT:\n"
             "    !python -m trilingua.evaluate --model {SFT} --data_dir {DATA} --out_dir {RES} --chat --tag sft --langid_n 30 --device cuda"),
    ("code", "from IPython.display import Image, Markdown\n"
             "for f in sorted(glob.glob(f'{RES}/*.png')): print(f); display(Image(f))\n"
             "display(Markdown(open(f'{RES}/samples_base.md').read()))"),
    ("md", "## Export to safetensors and push"),
    ("code", "!python -m trilingua.export --ckpt {BASE} --tokenizer {DATA}/tokenizer.json --out_dir /kaggle/working/export\n"
             "if SFT:\n"
             "    !python -m trilingua.export --ckpt {SFT} --tokenizer {DATA}/tokenizer.json --out_dir /kaggle/working/export_sft\n"
             "if os.environ.get('HF_TOKEN'):\n"
             "    !python -m trilingua.export --ckpt {BASE} --tokenizer {DATA}/tokenizer.json --out_dir /kaggle/working/export --push {HF_REPO}\n"
             "    if SFT:\n"
             "        !python -m trilingua.export --ckpt {SFT} --tokenizer {DATA}/tokenizer.json --out_dir /kaggle/working/export_sft --push {HF_REPO} --path_in_repo sft\n"
             "else:\n"
             "    print('HF_TOKEN secret not set: skipping push')"),
    ("code", "# CPU sanity check of the exported model (what the Gradio app does)\n"
             "!python generate.py --model /kaggle/working/export --lang fr --prompt \"Il était une fois\" --max_new_tokens 60 --seed 0\n"
             "if SFT:\n"
             "    !python generate.py --model /kaggle/working/export_sft --lang es --mode write --prompt \"los volcanes\" --max_new_tokens 80 --seed 0"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, cells in [("01_tokenizer_and_data", NB1), ("02_pretrain", NB2), ("03_sft", NB3),
                        ("04_eval_and_export", NB4)]:
        n = nb(cells)
        nbf.validate(n)
        nbf.write(n, os.path.join(OUT, f"{name}.ipynb"))
        print("wrote", name)


if __name__ == "__main__":
    main()
