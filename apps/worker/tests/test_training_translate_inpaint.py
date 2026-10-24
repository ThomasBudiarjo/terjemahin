"""Training system part 2: translator fine-tuning, AOT inpainter training/export, training CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WORKER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER / "training"))

import cli
import train_translator as tt

torch = pytest.importorskip("torch")
import export_inpaint_onnx as ex
import train_inpaint as ti

AOT_CKPT = WORKER / "weights" / "aot-inpainting.ckpt"
NLLB = "facebook/nllb-200-distilled-600M"


def _nllb_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(NLLB, local_files_only=True)
    except OSError:
        return None


# --- cli -------------------------------------------------------------------------------------


def test_cli_lists_and_dispatches(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for cmd in ("synth", "detector", "ocr", "translator", "inpaint", "pseudo-label", "export-inpaint"):
        assert cmd in out
    assert cli.main(["nope"]) == 2
    assert cli.main(["synth", "--help"]) == 0
    assert "--n" in capsys.readouterr().out
    assert sys.argv[0] != str(WORKER / "training" / "synth.py")  # argv restored


# --- translator ------------------------------------------------------------------------------


def test_metrics_and_data(tmp_path):
    assert tt.bleu(["aku mau makan nasi"], ["aku mau makan nasi"]) == pytest.approx(100.0)
    assert tt.bleu(["x y z"], ["aku mau makan nasi"]) == 0.0
    assert 0 < tt.bleu(["aku mau makan"], ["aku mau makan nasi"]) < 100
    (tmp_path / "a.jsonl").write_text('{"src": "え？", "tgt": "Eh?"}\n\n{"src": "s", "tgt": ""}\n')
    (tmp_path / "b.tsv").write_text("うん\tIya\nはい\tYa\n")
    assert tt.read_pairs(tmp_path / "a.jsonl") == [("え？", "Eh?")]
    assert tt.read_pairs(tmp_path / "b.tsv") == [("うん", "Iya"), ("はい", "Ya")]
    pairs = [(f"s{i % 7}", f"t{i}") for i in range(40)]
    tr, va = tt.split_pairs(pairs, 0.2, 0)
    assert va and not {s for s, _ in tr} & {s for s, _ in va}
    assert tt.lang_codes("nllb", "ja", "id") == ("jpn_Jpan", "ind_Latn")
    assert tt.lang_codes("m2m100", "ja", "id") == ("ja", "id")


def _tiny_nllb(tok, path: Path) -> Path:
    from transformers import M2M100Config, M2M100ForConditionalGeneration

    cfg = M2M100Config(
        vocab_size=len(tok), d_model=16, encoder_layers=1, decoder_layers=2,
        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
        decoder_ffn_dim=32, max_position_embeddings=128, pad_token_id=tok.pad_token_id,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.eos_token_id, scale_embedding=True,
    )
    torch.manual_seed(0)
    M2M100ForConditionalGeneration(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.mark.skipif(_nllb_tokenizer() is None, reason="NLLB tokenizer not in the HF cache")
def test_translator_finetune_roundtrip(tmp_path):
    """3 CPU steps on a tiny NLLB-shaped model; the saved dir loads in the worker translator."""
    from transformers import AutoModelForSeq2SeqLM

    from terjemahin_worker.translate.hf_seq2seq import Seq2SeqTranslator

    tok = _nllb_tokenizer()
    base = _tiny_nllb(tok, tmp_path / "base")
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    trainable, total = tt.freeze(model, False, False, 1)
    layer = sum(p.numel() for p in model.get_decoder().layers[-1].parameters())
    ln = sum(p.numel() for p in model.get_decoder().layer_norm.parameters())
    assert trainable == layer + ln < total
    tr_all, _ = tt.freeze(model, False, False, -1)
    emb = model.get_input_embeddings().weight.numel()
    assert tr_all == total - emb
    assert tt.freeze(model, True, False, 1)[0] == layer + ln + emb  # embeddings combine with N layers
    enc = sum(p.numel() for n, p in model.named_parameters() if ".encoder." in n and "embed" not in n)
    assert tt.freeze(model, False, True, -1)[0] == total - emb - enc

    data = tmp_path / "c.jsonl"
    data.write_text("\n".join(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) for s, t in [
        ("ありがとう", "Terima kasih"), ("待って！", "Tunggu!"), ("うるさいなぁ", "Berisik, ah"),
        ("行こう", "Ayo pergi"), ("本当に？", "Beneran?"), ("大丈夫", "Nggak apa-apa"),
    ]))
    out = tmp_path / "tuned"
    res = tt.main(["--data", str(data), "--model", str(base), "--out", str(out), "--steps", "3",
                   "--batch", "2", "--val-ratio", "0.34", "--decoder-layers", "1", "--lr", "1e-3",
                   "--eval-beams", "1", "--eval-every", "3"])
    assert res["steps"] == 3 and res["family"] == "nllb"
    assert [h["step"] for h in res["history"]] == [0, 3]
    assert {"chrf", "bleu"} <= res["history"][-1].keys()
    meta = json.loads((out / "terjemahin.json").read_text())
    assert meta["family"] == "nllb" and meta["opt"].startswith("translator.model=")
    assert (out / "config.json").exists() and (out / "tokenizer_config.json").exists()

    tuned = AutoModelForSeq2SeqLM.from_pretrained(out)  # stored float16, loads as float32
    assert tuned.get_input_embeddings().weight.dtype == torch.float32
    assert torch.allclose(tuned.get_input_embeddings().weight, model.get_input_embeddings().weight,
                          atol=1e-4)
    assert not torch.allclose(tuned.get_decoder().layers[-1].fc1.weight,
                              model.get_decoder().layers[-1].fc1.weight, atol=1e-4)
    assert torch.allclose(tuned.get_decoder().layers[0].fc1.weight,
                          model.get_decoder().layers[0].fc1.weight, atol=1e-4)
    val = tmp_path / "v.tsv"  # held-out file overlapping the data: its sources leave training
    val.write_text("ありがとう\tMakasih\n待って！\tTunggu!\n")
    res = tt.main(["--data", str(data), "--val", str(val), "--model", str(base), "--out",
                   str(tmp_path / "acc"), "--epochs", "1", "--batch", "1", "--grad-accum", "3",
                   "--eval-beams", "1", "--no-baseline", "--decoder-layers", "1"])
    assert (res["steps"], res["epochs"]) == (2, 1)  # 4 micro-batches = 3 + a stepped tail of 1

    outs = Seq2SeqTranslator(str(out), "nllb", num_beams=1, retries=0).translate_texts(
        ["ありがとう", "坂本くん？"], "ja", "id")
    assert len(outs) == 2 and all(isinstance(s, str) for s in outs)


# --- inpainting ------------------------------------------------------------------------------


def _write_triplets(root: Path, n: int = 3, size: int = 160) -> None:
    rng = np.random.default_rng(0)
    for k in ("clean", "mask", "texted"):
        (root / "inpaint" / k).mkdir(parents=True)
    yy, xx = np.mgrid[:size, :size]
    for i in range(n):
        tone = (((xx + i) // 4 + yy // 4) % 2 * 60 + 150).astype(np.uint8)
        clean = np.dstack([tone] * 3)
        cv2.circle(clean, (size // 2, size // 2), size // 3, (255, 255, 255), -1)
        mask = np.zeros((size, size), np.uint8)
        for _ in range(6):
            x, y = rng.integers(10, size - 30, 2)
            cv2.putText(mask, "AB", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        texted = clean.copy()
        texted[mask > 0] = 0
        stem = f"{i:06d}"
        cv2.imwrite(str(root / "inpaint" / "clean" / f"{stem}.png"), clean)
        cv2.imwrite(str(root / "inpaint" / "mask" / f"{stem}.png"), mask)
        cv2.imwrite(str(root / "inpaint" / "texted" / f"{stem}.png"), texted)


def test_generator_shapes_and_fold():
    net = ti.AOTGenerator(ch=8, blocks=1).eval()
    x, m = torch.rand(2, 3, 96, 72) * 2 - 1, (torch.rand(2, 1, 96, 72) > 0.7).float()
    y = net(x, m)
    assert y.shape == (2, 3, 96, 72) and float(y.abs().max()) <= 1
    folded = ex.fold_weight_standardization(_copy(net))
    assert not any(isinstance(mod, (ti.WSConv2d, ti.WSConvTranspose2d)) for mod in folded.modules())
    assert torch.allclose(folded(x, m), y, atol=1e-5)


def _copy(net):
    other = ti.AOTGenerator(**net.config).eval()
    other.load_state_dict(net.state_dict())
    return other


def test_random_text_mask_and_triplets(tmp_path):
    import random

    m = ti.random_text_mask(300, 200, random.Random(0))
    assert m.shape == (300, 200) and 0.001 < (m > 0).mean() < 0.6
    _write_triplets(tmp_path, n=2)
    trip = ti.find_triplets(tmp_path)
    assert len(trip) == 2 and ti.find_triplets(tmp_path / "inpaint") == trip
    with pytest.raises(FileNotFoundError):
        ti.find_triplets(tmp_path / "nothing")


def test_inpaint_train_export_and_run_through_lama(tmp_path):
    """Scratch tiny net: 3 steps -> best.pt -> ONNX -> the worker's LamaInpainter runs it."""
    from terjemahin_worker.inpaint.lama import LamaInpainter

    _write_triplets(tmp_path / "data")
    onnx_path = tmp_path / "aot.onnx"
    res = ti.main(["--data", str(tmp_path / "data"), "--init", "none", "--ch", "8", "--blocks", "1",
                   "--size", "96", "--batch", "2", "--steps", "3", "--eval-every", "3",
                   "--val-crops", "2", "--warmup", "1", "--out", str(tmp_path / "run"),
                   "--export", str(onnx_path)])
    assert [h["step"] for h in res["history"]] == [0, 3]
    assert np.isfinite(res["history"][-1]["psnr_hole"]) and onnx_path.exists()
    ck = torch.load(tmp_path / "run" / "best.pt", weights_only=False)
    assert ck["config"] == {"ch": 8, "blocks": 1}

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    odd = {"image": np.random.rand(1, 3, 101, 99).astype(np.float32),
           "mask": np.zeros((1, 1, 101, 99), np.float32)}
    odd["mask"][..., 40:60, 30:50] = 1
    y = sess.run(None, odd)[0]  # H, W not multiples of 4: padded inside the graph
    assert y.shape == (1, 3, 101, 99)
    keep = odd["mask"][:, 0] == 0
    assert np.allclose(y.transpose(1, 0, 2, 3)[:, keep], odd["image"].transpose(1, 0, 2, 3)[:, keep])

    from terjemahin_worker.config import get_settings
    from terjemahin_worker.registry import INPAINTERS

    factory = INPAINTERS.get("lama_manga").factory  # `--opt inpainter.weights=` reaches lama.py
    assert factory(get_settings(), weights=str(onnx_path)).model == onnx_path
    with pytest.raises(FileNotFoundError):
        factory(get_settings(), weights=str(tmp_path / "missing.onnx"))

    img = np.full((200, 240, 3), 255, np.uint8)
    img[::6] = 40
    mask = np.zeros((200, 240), np.uint8)
    mask[80:120, 100:140] = 255
    out = LamaInpainter(onnx_path, out_scale=1.0, flat_tol=0, dilate=0, feather=0).inpaint(img, mask)
    assert out.shape == img.shape
    assert np.array_equal(out[mask == 0], img[mask == 0])


def test_inpaint_eval_ignores_holeless_val_pages(tmp_path):
    """A held-out page without holes must not yield a perfect score that pins best.pt to step 0."""
    _write_triplets(tmp_path / "data")
    cv2.imwrite(str(tmp_path / "data" / "inpaint" / "mask" / "000000.png"), np.zeros((160, 160), np.uint8))
    res = ti.main(["--data", str(tmp_path / "data"), "--init", "none", "--ch", "8", "--blocks", "1",
                   "--size", "96", "--batch", "2", "--steps", "2", "--eval-every", "1",
                   "--val-ratio", "0.34", "--val-crops", "2", "--warmup", "1",
                   "--out", str(tmp_path / "run")])
    assert all(h["hole_px"] > 0 and h["psnr_hole"] < 80 for h in res["history"])
    assert ti.evaluate(ti.AOTGenerator(ch=8, blocks=1), [])["psnr_hole"] is None


@pytest.mark.skipif(not AOT_CKPT.exists(), reason="weights/aot-inpainting.ckpt not downloaded")
def test_released_aot_weights_export_and_beat_telea(tmp_path):
    """The released checkpoint loads strictly into our port and, exported, fills text-column holes
    in hatching far closer to the truth than cv2 Telea does (measured: MAE 57 vs 84)."""
    from terjemahin_worker.inpaint.lama import LamaInpainter

    onnx_path = ex.export("aot", tmp_path / "aot.onnx", size=128)
    yy, xx = np.mgrid[:256, :256]
    clean = np.dstack([np.where((xx + yy) % 10 < 3, 40, 230).astype(np.uint8)] * 3)
    mask = np.zeros((256, 256), np.uint8)
    for x in range(90, 170, 22):
        mask[70:190, x : x + 10] = 255
    texted = clean.copy()
    texted[mask > 0] = 0
    out = LamaInpainter(onnx_path, out_scale=1.0, flat_tol=0, dilate=0, feather=0).inpaint(texted, mask)
    hole = mask > 0
    mae = np.abs(out[hole].astype(float) - clean[hole]).mean()
    telea = cv2.inpaint(texted, mask, 3, cv2.INPAINT_TELEA)
    mae_telea = np.abs(telea[hole].astype(float) - clean[hole]).mean()
    assert mae < 0.8 * mae_telea, (mae, mae_telea)
