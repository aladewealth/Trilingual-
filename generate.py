#!/usr/bin/env python
"""Command-line generation with streaming output.

Examples:
  python generate.py --model aladewealth/trilingua --lang fr --prompt "Il était une fois"
  python generate.py --model export --lang es --mode write --prompt "los volcanes de México"
  python generate.py --model out/ckpt_latest.pt --tokenizer data/tokenizer.json --lang en --prompt "The"
"""
import argparse
import sys

import torch

from trilingua.inference import build_prompt, load_bundle, stream_text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="aladewealth/trilingua", help="HF repo id, exported dir, or .pt checkpoint")
    p.add_argument("--subfolder", default=None, help="e.g. 'sft' inside the HF repo")
    p.add_argument("--tokenizer", default=None, help="needed with a raw .pt checkpoint")
    p.add_argument("--lang", default="en", choices=["en", "es", "fr"])
    p.add_argument("--mode", default="continue", choices=["continue", "write", "chat"])
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no_stream", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args(argv)
    torch.set_grad_enabled(False)
    b = load_bundle(a.model, a.tokenizer, a.subfolder, a.device)
    ids = build_prompt(b, a.lang, a.mode, a.prompt)
    printed = ""
    if a.mode == "continue":
        sys.stdout.write(a.prompt)
    final = ""
    for text in stream_text(b, ids, a.max_new_tokens, a.temperature, a.top_k, a.top_p,
                            a.repetition_penalty, a.seed):
        final = text
        if not a.no_stream and text.startswith(printed):
            sys.stdout.write(text[len(printed):]); sys.stdout.flush()
            printed = text
    if a.no_stream:
        sys.stdout.write(final)
    elif final.startswith(printed):
        sys.stdout.write(final[len(printed):])
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
