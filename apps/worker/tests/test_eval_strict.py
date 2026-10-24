"""Stricter harness checks: partial detections and meaning conflicts the embedding misses."""

from pathlib import Path

from terjemahin_worker.eval import bench as B
from terjemahin_worker.eval import lines as L
from terjemahin_worker.eval import metrics as M

WORKER = Path(__file__).resolve().parent.parent


def test_a_box_that_reads_part_of_its_bubble_is_partial():
    # judge 1: はーい read as はー, ドクン as クン, じゃあいつは……？ as あいつは
    s = M.match_detections(["はー", "クン", "あいつは", "ドクシ", "なに．．．"],
                           ["はーい", "ドクン", "じゃあいつは……？", "ドクン", "なに…？"])
    assert len(s.matches) == 5
    assert sorted(m.gt for m in s.partial) == [0, 1, 2]  # a misread (ドクシ) is no cut
    gt = B.GroundTruth(["はーい", "ドクン"], ["", ""])
    d, _ = B.score_detection(["はー", "ドクン"], gt)
    assert d["matched"] == 2 and d["partial"] == 1 and d["recall"] == 0.5
    assert B.checks({"detection": {**d, "false_positives": 0}})["detect"] is False


def test_coverage():
    assert M.coverage("はー", "はーい") == 0.5
    assert M.coverage("ドクンドクン", "ドクン") == 1.0  # longer reads are not cut off
    assert M.coverage("でもでもね", "でもねでもね") > M.PARTIAL_COVERAGE  # one char dropped


def test_meaning_conflicts_catch_what_cosine_accepts():
    # judge 2's pairs that MiniLM scored >= 0.8
    assert "roles" in M.meaning_conflicts("Kau tidak akan pernah memaafkanku!",
                                          "Aku nggak akan pernah memaafkanmu!")
    assert "voice" in M.meaning_conflicts("Aku marah pada guru", "Aku dimarahi guru")
    assert "participants" in M.meaning_conflicts("Aku ingin kau tidur",
                                                 "Berisik ah, biarin aku tidur dong")
    assert "participants" in M.meaning_conflicts("Aku menyukainya!", "Senpai, aku suka kamu!")
    assert "negation" in M.meaning_conflicts("Aku bisa lari", "Aku nggak bisa lari")


def test_meaning_conflicts_leave_paraphrases_alone():
    ok = [
        ("Hei, hei, apa yang kau pikirkan tentang Sakamoto?",
         "Hei, hei, Mei, menurutmu Sakamoto itu gimana?"),
        ("Eh, itu saja?", "Eh, cuma itu?"),
        ("A-aku... tidak seperti itu...", "A-aku sih... nggak gimana-gimana..."),
        ("Aku tak akan pernah memaafkanmu!", "Nggak akan pernah kumaafkan!"),  # implied object
        ("Dia menggunakan ucapan sopan bahkan dengan staf toko, itu mengesankan, gitu",
         "Dia sopan pakai bahasa halus bahkan ke pelayan toko, keren gitu"),
        ("Reaksinya selalu menarik", "Reaksi dia selalu lucu"),
        ("Guru memarahiku", "Aku dimarahi guru"),  # active for passive, same roles: allowed
    ]
    for hyp, ref in ok[:-1]:
        assert M.meaning_conflicts(hyp, ref) == [], (hyp, ref)
    assert "roles" not in M.meaning_conflicts(*ok[-1])


def test_persons():
    assert M.persons("Kau tidak akan pernah memaafkanku") == [2, 1]
    assert M.persons("Harus berapa kali kubilang baru kamu ngerti?") == [1, 2]
    assert M.persons("Buku itu ilmu") == []
    assert M.persons("Aku membuatmu menunggu", standalone=True) == [1]


def test_heldout_line_set_loads():
    lines = L.load_lines(WORKER / "eval" / "fixtures" / "lines_ja_id.tsv")
    assert len(lines) >= 30 and all(ln.ja and ln.ref for ln in lines)
    res = L.score([ln.ref for ln in lines], lines, 0.8)
    # every reference is adequate against itself (no self-conflict, cosine 1 when available)
    assert res["conflicts"] == 0
    if res["semantic"]:
        assert res["adequate"] == len(lines)


def test_render_check_reads_quarter_turned_text_upright():
    import cv2
    import numpy as np

    from terjemahin_worker.models import BBox, Bubble, TextRegion

    page = np.full((300, 100, 3), 255, np.uint8)
    final = page.copy()
    layer = np.full((40, 260, 3), 255, np.uint8)
    cv2.putText(layer, "HELLO WORLD", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    final[20:280, 30:70] = cv2.rotate(layer, cv2.ROTATE_90_CLOCKWISE)  # reads top to bottom
    b = Bubble(region=TextRegion(id=0, bbox=BBox(x1=30, y1=20, x2=70, y2=280)),
               source_text="x", target_text="HELLO WORLD")

    class P:  # what render_check reads off a renderer plan
        bubble, angle, layout = b, -90.0, None

    seen = []

    def reader(canvas, quads):
        seen.append(canvas.shape[:2])
        return ["HELLO WORLD"] * len(quads)

    res, _ = M.render_check(page, final, page, [b], plans=[P()], reader=reader)
    assert seen[-1][1] > seen[-1][0]  # the last read was on an upright (wide) crop
    assert res[0].read_cer == 0.0
