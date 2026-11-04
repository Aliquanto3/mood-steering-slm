"""Pic 1.4 : format des paires et logique pure du balayage, sans modèle ni téléchargement."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibration_scan", Path(__file__).parent.parent / "spikes" / "calibration_scan.py"
)
cs = importlib.util.module_from_spec(_SPEC)
sys.modules["calibration_scan"] = cs
_SPEC.loader.exec_module(cs)


def _write(path: Path, rows: list) -> Path:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


# --- paires -----------------------------------------------------------------------------------


def test_versioned_pairs_format():
    lines = cs.PAIRS_PATH.read_text(encoding="utf-8").splitlines()
    assert all(set(json.loads(x)) == {"positive", "negative"} for x in lines)
    pairs = cs.load_pairs()
    assert len(pairs) == len(lines) == 100
    texts = [t for p in pairs for t in p]
    assert len(set(texts)) == 200


def test_load_pairs_rejects_extra_key(tmp_path):
    p = _write(tmp_path / "p.jsonl", [{"positive": "a", "negative": "b", "x": 1}])
    with pytest.raises(ValueError, match="ligne 1 : clés"):
        cs.load_pairs(p)


def test_load_pairs_rejects_duplicate(tmp_path):
    rows = [{"positive": "a", "negative": "b"}, {"positive": "c", "negative": "a"}]
    with pytest.raises(ValueError, match="ligne 2 : texte en double"):
        cs.load_pairs(_write(tmp_path / "p.jsonl", rows))


def test_load_pairs_rejects_empty_and_identical(tmp_path):
    with pytest.raises(ValueError, match="vides"):
        cs.load_pairs(_write(tmp_path / "a.jsonl", [{"positive": " ", "negative": "b"}]))
    with pytest.raises(ValueError, match="identiques"):
        cs.load_pairs(_write(tmp_path / "b.jsonl", [{"positive": "a", "negative": "a"}]))


def test_load_pairs_skips_blank_lines(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"positive": "a", "negative": "b"}\n\n', encoding="utf-8")
    assert cs.load_pairs(p) == [("a", "b")]


# --- score de ton, cohérence, langue ---------------------------------------------------------


def test_tone_score_values():
    assert cs.tone_score("Le train est arrivé en gare.") == 0.0
    assert cs.tone_score("Quel bonheur, je suis heureux !") == pytest.approx(2 / 3)
    assert cs.tone_score("Une profonde tristesse, des larmes.") == pytest.approx(-2 / 3)
    assert cs.tone_score("Joie et chagrin.") == 0.0
    # Mots proches qui ne doivent pas compter : « seulement », « gain », « beaucoup ».
    assert cs.tone_score("Seulement un gain de beaucoup.") == 0.0


def test_is_coherent():
    ok = "Ce matin, j'ai ouvert les volets et le soleil entrait doucement dans la chambre."
    assert cs.is_coherent(ok)
    assert not cs.is_coherent("Trop court.")
    assert not cs.is_coherent("Let's let! " * 10)
    assert not cs.is_coherent("締!<!<締!<!< 締!<!< ab cd ef gh ij kl mn")
    # Trigramme répété trois fois : boucle.
    loop = "il fait beau et il fait beau et il fait beau et puis rien"
    assert not cs.is_coherent(loop)


def test_is_french():
    assert cs.is_french("Je suis allé dans le jardin avec mon frère.")
    assert not cs.is_french("Today is a fantastic day and I was happy with it.")
    assert not cs.is_french("Bonjour")


# --- couches et plage utile ------------------------------------------------------------------


def test_layer_range():
    assert cs.layer_range(24, (0.0, 1.0)) == (1, 23)
    assert cs.layer_range(24, (0.25, 0.75)) == (6, 18)
    assert cs.layer_range(36, (0.5, 0.85)) == (18, 31)
    assert cs.layer_range(4, (0.9, 0.95)) == (3, 3)


def _stats(points):
    return [cs.DoseStat(d, t, c, 3) for d, t, c in points]


def test_aggregate_groups_and_counts_french_coherent():
    rows = [
        {"layers": "all", "dose": 0.1, "tone": 0.5, "coherent": True, "french": True},
        {"layers": "all", "dose": 0.1, "tone": 0.0, "coherent": True, "french": False},
        {"layers": "all", "dose": 0.0, "tone": 0.0, "coherent": True, "french": True},
        {"layers": "mid", "dose": 0.1, "tone": 1.0, "coherent": False, "french": True},
    ]
    agg = cs.aggregate(rows)
    assert agg["all"] == [cs.DoseStat(0.0, 0.0, 1.0, 1), cs.DoseStat(0.1, 0.25, 0.5, 2)]
    assert agg["mid"] == [cs.DoseStat(0.1, 1.0, 0.0, 1)]


def test_useful_range_longest_contiguous_run():
    stats = _stats(
        [
            (0.0, 0.0, 1.0),
            (0.05, 0.1, 1.0),  # ton trop proche de la base
            (0.1, 0.3, 1.0),
            (0.15, 0.5, 1.0),
            (0.2, 0.6, 1.0),
            (0.3, 0.8, 0.0),  # incohérent : coupe la suite
            (0.4, 0.9, 1.0),
        ]
    )
    rng = cs.useful_range(stats, 1)
    assert rng == (0.1, 0.2)
    assert cs.meets_ratio(rng)


def test_useful_range_relative_to_base_and_negative_side():
    stats = _stats([(0.0, 0.2, 1.0), (0.1, 0.4, 1.0), (-0.1, -0.1, 1.0), (-0.2, -0.3, 2 / 3)])
    # +0,1 ne s'écarte que de 0,2 de la base : pas utile.
    assert cs.useful_range(stats, 1) is None
    assert cs.useful_range(stats, -1) == (0.1, 0.2)
    # Juste sous le seuil de cohérence : refusé.
    assert cs.useful_range(_stats([(-0.2, -0.5, 0.66)]), -1) is None


def test_meets_ratio_bounds():
    assert cs.meets_ratio((0.1, 0.15))
    assert not cs.meets_ratio((0.1, 0.149))
    assert not cs.meets_ratio(None)


def test_tone_score_emojis_and_false_friends():
    assert cs.tone_score("Réunion reportée 🎉🥳") == pytest.approx(2 / 3)
    assert cs.tone_score("Il pleut 😢") == pytest.approx(-0.5)
    # « pain » est le pain, pas la douleur anglaise.
    assert cs.tone_score("J'ai acheté du pain.") == 0.0


def test_chat_prompt():
    q = "Raconte ta matinée."
    assert cs.chat_prompt("qwen35", q) == (
        "<|im_start|>user\nRaconte ta matinée.<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    assert cs.chat_prompt("llama", q).endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert q in cs.chat_prompt("llama", q)
    with pytest.raises(ValueError, match="gemma3"):
        cs.chat_prompt("gemma3", q)
