import json
import os

import torch

from trilingua import train as T


def _args(d, out, extra=()):
    cfg = os.path.join(os.path.dirname(out), "model.json")
    json.dump(dict(vocab_size=300, max_seq_len=32, d_model=32, n_layer=2, n_head=2), open(cfg, "w"))
    return ["--model", cfg, "--seq_len", "32", "--data_dir", d, "--micro_batch", "4",
            "--global_batch_tokens", "256", "--max_steps", "8", "--warmup_steps", "2", "--lr", "3e-3",
            "--eval_interval", "4", "--eval_chunks", "4", "--ckpt_interval", "100", "--log_interval", "1",
            "--sample_tokens", "0", "--device", "cpu", "--dtype", "fp32", "--compile", "off",
            "--out_dir", out, *extra]


def _final_state(out):
    return torch.load(os.path.join(out, "ckpt_latest.pt"), map_location="cpu", weights_only=False)


def test_resume_is_bitwise_equivalent(synth_data, tmp_path):
    d, _ = synth_data
    a_out, b_out = str(tmp_path / "a"), str(tmp_path / "b")
    torch.manual_seed(0)
    T.main(_args(d, a_out))                                  # 8 steps straight
    T.main(_args(d, b_out, ["--stop_step", "3"]))            # 3 steps, checkpoint, exit
    assert _final_state(b_out)["step"] == 3
    T.main(_args(d, b_out))                                  # auto-resume 3 -> 8
    A, B = _final_state(a_out), _final_state(b_out)
    assert A["step"] == B["step"] == 8
    for k in A["model"]:
        assert torch.equal(A["model"][k], B["model"][k]), k
    sa, sb = A["optimizer"]["state"], B["optimizer"]["state"]
    for k in sa:
        assert torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"])
    assert A["loader"]["cursor"] == B["loader"]["cursor"]
    la = open(os.path.join(a_out, "train_log.csv")).read().splitlines()
    lb = open(os.path.join(b_out, "train_log.csv")).read().splitlines()
    col = la[0].split(",").index("loss")
    assert [r.split(",")[col] for r in la[1:]] == [r.split(",")[col] for r in lb[1:]]


def test_resume_from_other_dir_carries_logs(synth_data, tmp_path):
    d, _ = synth_data
    s1, s2 = str(tmp_path / "session1"), str(tmp_path / "session2")
    T.main(_args(d, s1, ["--stop_step", "4"]))
    T.main(_args(d, s2, ["--resume_dirs", s1]))
    assert _final_state(s2)["step"] == 8
    rows = open(os.path.join(s2, "train_log.csv")).read().splitlines()[1:]
    assert [int(r.split(",")[0]) for r in rows] == list(range(1, 9))


def test_loss_decreases(synth_data, tmp_path):
    d, _ = synth_data
    out = str(tmp_path / "c")
    T.main(_args(d, out, ["--max_steps", "30", "--eval_interval", "30"]))
    rows = open(os.path.join(out, "train_log.csv")).read().splitlines()
    col = rows[0].split(",").index("loss")
    losses = [float(r.split(",")[col]) for r in rows[1:]]
    assert losses[-1] < losses[0] - 0.3
