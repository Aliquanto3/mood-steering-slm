"""Tests du banc H6 (story 5.1) : moteur et classifieur simulés, aucun modèle chargé."""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from mood_steering.steering import Steerer

_SPEC = importlib.util.spec_from_file_location(
    "run_pulse", Path(__file__).parent.parent / "scripts" / "run_pulse.py"
)
rp = importlib.util.module_from_spec(_SPEC)
sys.modules["run_pulse"] = rp
_SPEC.loader.exec_module(rp)

N_EMBD, N_LAYER = 4, 6


class FakeBackend:
    """Mime llama.cpp : un fragment par token, « joie » quand un vecteur est posé."""

    n_embd = N_EMBD
    n_layer = N_LAYER

    def __init__(self, length=192):
        self.length = length
        self.cvec = None
        self.calls = []

    def reset(self):
        pass

    def set_cvec(self, data, n_embd, il_start, il_end):
        self.cvec = None if data is None else float(data[0])

    def generate(self, prompt, system, max_tokens, seed):
        return "".join(self.stream(prompt, system, max_tokens, seed))

    def stream(self, prompt, system, max_tokens, seed):
        return self.stream_scheduled(prompt, system, max_tokens, seed, lambda t: None)

    def stream_scheduled(self, prompt, system, max_tokens, seed, on_token_start):
        self.calls.append(prompt)
        for t in range(max_tokens):
            on_token_start(t)
            if t >= self.length:  # fin de génération anticipée
                return
            yield "joie " if self.cvec else "plat "


def make_steerer(length=192):
    vector = np.ones((N_LAYER - 1, N_EMBD), dtype=np.float32)
    backend = FakeBackend(length)
    return Steerer(backend, vector, max_scale=1.0, il_start=1, il_end=3), backend


class FakeScorer:
    """Ton = (mots « joie » - mots « plat ») / mots."""

    def score(self, texts):
        return [(t.count("joie") - t.count("plat")) / max(1, len(t.split())) for t in texts]


def prompts(n):
    return [(it, k) for k, it in enumerate(rp.pulse_prompts(n))]


def test_pulse_prompts():
    items = rp.pulse_prompts(50)
    assert [i.id for i in items[:2]] == ["pulse000", "pulse001"]
    assert len({i.prompt for i in items}) == 50
    assert "{s}" not in items[0].prompt
    assert rp.pulse_prompts(3) == items[:3]
    for bad in (0, 51):
        with pytest.raises(ValueError, match="n_prompts"):
            rp.pulse_prompts(bad)


def test_profils_des_bras():
    assert rp.arm_schedule("neutral")(100) == 0.0
    assert rp.arm_schedule("constant")(0) == rp.arm_schedule("constant")(191) == 1.0
    pulse = rp.arm_schedule("pulse")
    assert pulse(0) == 0.0 and pulse(48) == 1.0
    assert pulse(80) == pytest.approx(0.5)  # une demi-vie après le pic
    assert pulse(47) < 1.0 and pulse(49) < 1.0
    with pytest.raises(ValueError, match="bras inconnu"):
        rp.arm_schedule("x")


def test_generate_row_enregistre_fragments_tokens_et_doses():
    steerer, _ = make_steerer()
    item = rp.pulse_prompts(1)[0]
    row = rp.generate_row(steerer, "pulse", item, 7, 192)
    assert row["arm"] == "pulse" and row["seed"] == 7 and row["item_id"] == "pulse000"
    assert row["n_tokens"] == 192 and len(row["doses"]) == 192
    assert row["tok"] == list(range(192))
    assert row["doses"][0] == 0.0 and row["doses"][48] == 1.0
    assert row["output"] == "".join(row["pieces"])
    # neutre : jamais de vecteur ; constant : vecteur dès le token 0
    assert rp.generate_row(steerer, "neutral", item, 0, 192)["output"] == "plat " * 192
    assert rp.generate_row(steerer, "constant", item, 0, 192)["output"] == "joie " * 192
    # pulse : dose 0 au token 0 (pas de vecteur), puis vecteur posé
    assert row["pieces"][0] == "plat " and row["pieces"][1] == "joie "


def test_n_tokens_exclut_le_token_de_fin():
    steerer, _ = make_steerer(length=70)
    row = rp.generate_row(steerer, "constant", rp.pulse_prompts(1)[0], 0, 192)
    assert row["n_tokens"] == 70 and len(row["pieces"]) == 70


def test_segment_texts_complets_seulement():
    row = {
        "n_tokens": 70,
        "pieces": [f"w{i} " for i in range(70)],
        "tok": list(range(70)),
    }
    segs = rp.segment_texts(row)
    assert len(segs) == 6
    assert segs[0] == "".join(f"w{i} " for i in range(32))
    assert segs[1] == "".join(f"w{i} " for i in range(32, 64))
    assert segs[2:] == [None] * 4  # 70 tokens : deux segments complets seulement


def test_segment_texts_indices_de_token_pas_de_fragments():
    # des tokens sans texte (octets UTF-8 partiels) ne décalent pas les segments
    pieces = ["a ", "b ", "c "]
    row = {"n_tokens": 192, "pieces": pieces, "tok": [0, 40, 100]}
    segs = rp.segment_texts(row)
    assert segs == ["a ", "b ", None, "c ", None, None]


def test_mean_doses():
    rows = [
        {"n_tokens": 192, "doses": [1.0] * 192},
        {"n_tokens": 40, "doses": [0.0] * 41},
    ]
    doses = rp.mean_doses(rows)
    assert doses[0] == pytest.approx(0.5)  # les deux sorties atteignent le segment 0
    assert doses[1] == 1.0  # seule la première atteint le segment 1
    assert doses[5] == 1.0
    assert rp.mean_doses([{"n_tokens": 10, "doses": [0.0] * 11}]) == [None] * 6


def test_run_campaign_ecrit_trois_bras_et_reprend(tmp_path):
    steerer, backend = make_steerer()
    items = prompts(2)
    assert rp.run_campaign(steerer, items, tmp_path)
    for arm in rp.ARMS:
        rows = rp.load_rows(rp.arm_file(tmp_path, arm))
        assert [r["item_id"] for r in rows] == ["pulse000", "pulse001"]
    assert len(backend.calls) == 6
    # reprise : rien n'est régénéré
    assert rp.run_campaign(steerer, items, tmp_path)
    assert len(backend.calls) == 6


def test_run_campaign_reprend_apres_ligne_tronquee_et_agrandissement(tmp_path):
    steerer, backend = make_steerer()
    assert rp.run_campaign(steerer, prompts(1), tmp_path)
    path = rp.arm_file(tmp_path, "pulse")
    path.write_bytes(path.read_bytes() + b'{"arm": "pul')  # arrêt brutal en cours d'écriture
    assert rp.run_campaign(steerer, prompts(2), tmp_path)
    rows = rp.load_rows(path)
    assert [r["item_id"] for r in rows] == ["pulse000", "pulse001"]
    assert len(backend.calls) == 3 + 3  # le premier prompt n'a pas été refait
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_run_campaign_arret_a_l_echeance(tmp_path):
    steerer, backend = make_steerer()
    done = rp.run_campaign(steerer, prompts(2), tmp_path, deadline=time.monotonic() - 1)
    assert done is False and backend.calls == []
    assert rp.load_rows(rp.arm_file(tmp_path, "neutral")) == []


def test_manifeste(tmp_path):
    cfg = rp.Config(n_prompts=3)
    manifest = rp.build_manifest(cfg, {"sha256": "abc"})
    assert manifest["rise"] == 48 and manifest["half_life"] == 32.0
    assert manifest["max_scale"] == 0.7 and (manifest["il_start"], manifest["il_end"]) == (8, 24)
    rp.check_manifest(tmp_path, manifest, resume=False)
    with pytest.raises(FileExistsError, match="--resume"):
        rp.check_manifest(tmp_path, manifest, resume=False)
    rp.check_manifest(tmp_path, manifest, resume=True)
    other = rp.build_manifest(rp.Config(n_prompts=4), {"sha256": "abc"})
    with pytest.raises(ValueError, match="configuration différente"):
        rp.check_manifest(tmp_path, other, resume=True)


def row(arm, item_id, tones, n_tokens=192):
    """Sortie synthétique : segment k = « joie » x tones[k] mots sur 8 (le reste « plat »)."""
    pieces, tok = [], []
    for k in range(6):
        for j in range(32):
            pieces.append("joie " if j < tones[k] else "plat ")
            tok.append(k * 32 + j)
    return {
        "arm": arm,
        "item_id": item_id,
        "pieces": pieces,
        "tok": tok,
        "n_tokens": n_tokens,
        "doses": [0.0] * (n_tokens + 1),
        "output": "".join(pieces),
    }


def test_analyse_criteres_exacts():
    # ton d'un segment = (2 x nb_joie - 32) / 32 ; nb_joie = 32 -> +1, 0 -> -1
    pulse_tones = [8, 32, 24, 16, 8, 0]  # monte, culmine au segment 1, redescend
    const_tones = [32, 32, 32, 32, 32, 32]
    neutral_tones = [0, 0, 0, 0, 0, 0]
    rows = {
        "neutral": [row("neutral", f"p{i}", neutral_tones) for i in range(4)],
        "constant": [row("constant", f"p{i}", const_tones) for i in range(4)],
        "pulse": [row("pulse", f"p{i}", pulse_tones) for i in range(4)],
    }
    s = rp.analyse(rows, FakeScorer(), n_boot=50, seed=0)
    # FakeScorer : (joie - plat) / mots, mots = 32
    assert s["cells"]["neutral"][0].mean == -1.0 and s["cells"]["neutral"][0].n == 4
    assert s["cells"]["constant"][5].mean == 1.0
    assert s["cells"]["pulse"][1].mean == 1.0 and s["cells"]["pulse"][5].mean == -1.0
    assert s["cells"]["pulse"][0].mean == pytest.approx((2 * 8 - 32) / 32)
    c = s["criteria"]
    assert c["pulse_pic_moins_dernier"].mean == 2.0 and c["pulse_pic_moins_dernier"].low == 2.0
    assert c["pulse_pic_moins_debut"].mean == pytest.approx(1.5)
    assert c["constant_moins_pulse_dernier"].mean == 2.0
    assert c["constant_pic_moins_dernier"].mean == 0.0
    # écart au neutre au segment 0 : constant 2.0, pulse 0.5 -> 1.5
    assert c["ecart_neutre_constant_moins_pulse_debut"].mean == pytest.approx(1.5)
    assert s["h6"] == {
        "pic_au_dessus_du_dernier": True,
        "pic_au_dessus_du_debut": True,
        "debut_plus_proche_du_neutre": True,
        "lecture": "oui",
    }


def test_analyse_lecture_non_si_le_pic_ne_depasse_pas_le_dernier():
    flat = [16] * 6
    rows = {a: [row(a, f"p{i}", flat) for i in range(3)] for a in rp.ARMS}
    s = rp.analyse(rows, FakeScorer(), n_boot=50)
    assert s["criteria"]["pulse_pic_moins_dernier"].mean == 0.0
    assert s["h6"]["pic_au_dessus_du_dernier"] is False
    assert s["h6"]["debut_plus_proche_du_neutre"] is False
    assert s["h6"]["lecture"] == "non"


def test_analyse_lecture_oui_mais_sans_contraste_de_debut():
    # le constant a le même profil que le pulse : montée et redescente lisibles, début identique
    tones = [8, 32, 24, 16, 8, 0]
    rows = {a: [row(a, f"p{i}", tones) for i in range(3)] for a in rp.ARMS}
    s = rp.analyse(rows, FakeScorer(), n_boot=50)
    assert s["h6"]["pic_au_dessus_du_dernier"] and s["h6"]["pic_au_dessus_du_debut"]
    assert s["h6"]["debut_plus_proche_du_neutre"] is False
    assert s["h6"]["lecture"] == "oui, mais"


def test_analyse_lecture_non_sans_montee():
    # redescente seule (le ton part du pic) : pas de montée, donc pas de profil lisible
    tones = [32, 32, 24, 16, 8, 0]
    rows = {a: [row(a, f"p{i}", tones) for i in range(3)] for a in rp.ARMS}
    s = rp.analyse(rows, FakeScorer(), n_boot=50)
    assert s["h6"]["pic_au_dessus_du_dernier"] is True
    assert s["h6"]["pic_au_dessus_du_debut"] is False
    assert s["h6"]["lecture"] == "non"


def test_analyse_sorties_courtes_et_appariement():
    tones = [32, 32, 0, 0, 0, 0]
    short = row("pulse", "p0", tones, n_tokens=70)  # n'atteint pas le dernier segment
    full = row("pulse", "p1", tones)
    rows = {
        "neutral": [row("neutral", f"p{i}", tones) for i in range(2)],
        "constant": [row("constant", f"p{i}", tones) for i in range(2)],
        "pulse": [short, full],
    }
    s = rp.analyse(rows, FakeScorer(), n_boot=50)
    assert s["cells"]["pulse"][5].n == 1 and s["cells"]["pulse"][1].n == 2
    # l'écart pic - dernier n'est calculé que sur le prompt qui a les deux segments
    assert s["criteria"]["pulse_pic_moins_dernier"].n == 1
    assert s["rates"]["pulse"]["short"] == 0.5
    assert s["rates"]["constant"]["short"] == 0.0


def test_taux_de_sorties_degenerees():
    good = (
        "Le marché du samedi matin ouvre tôt, les étals se remplissent de fruits colorés, "
        "de fromages, de pains chauds et chacun prend le temps de discuter avec les voisins."
    )
    rows = [
        {"output": good, "n_tokens": 192},
        {"output": "ha " * 60, "n_tokens": 192},  # boucle
        {"output": "oui", "n_tokens": 12},  # trop court
        {"output": good + " �", "n_tokens": 192},  # caractère de remplacement
    ]
    r = rp.output_rates(rows)
    assert r["n"] == 4 and r["degenerate"] == 0.75 and r["short"] == 0.25
    assert r["refusal"] == 0.0


def test_rapport_tableau_et_lecture_conservee(tmp_path):
    rows = {a: [row(a, f"p{i}", [8, 32, 24, 16, 8, 0]) for i in range(3)] for a in rp.ARMS}
    s = rp.analyse(rows, FakeScorer(), n_boot=50)
    md = rp.render_report(s, {"device": "GPU", "llama_cpp": "0.3.36", "git": "abc1234"})
    assert md.startswith("# H6")
    assert "| pulse | " in md and "seg 5 (160-191)" in md
    assert "(n=3)" in md and "GPU" in md and "abc1234" in md
    assert "-0.50 [" in md  # ton du segment 0 : (2 x 8 - 32) / 32
    assert "## Limites" in md and "## Sorties dégénérées" in md
    assert "<!-- lecture -->" in md  # gabarit à rédiger
    # une lecture rédigée à la main survit à la régénération
    path = tmp_path / "rapport.md"
    custom = "<!-- lecture -->\nMa lecture.\n<!-- /lecture -->"
    path.write_text(
        md.replace("(à rédiger après lecture du tableau)", "Ma lecture."), encoding="utf-8"
    )
    kept = rp.rb.preserved_lecture(path)
    assert kept == custom
    assert custom in rp.render_report(s, {}, kept)


def test_fmt_iv():
    assert rp.fmt_iv(None) == "n/d"
    iv = rp.Interval(mean=0.5, low=0.25, high=0.75, n=10)
    assert rp.fmt_iv(iv) == "+0.50 [+0.25 ; +0.75] (n=10)"


def test_arguments_par_defaut():
    args = rp.parse_args([])
    assert args.out == Path("data/runs/pulse-5-1") and args.n_prompts == 50
    assert args.report == Path("docs/rapport-pharmacocinetique.md")
    assert args.n_gpu_layers == 0 and args.resume is False


def test_main_vecteur_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="vecteur absent"):
        rp.main(["--out", str(tmp_path / "o"), "--vectors-dir", str(tmp_path)])
    assert not (tmp_path / "o").exists()


def test_main_rapport_sans_modele_quand_la_campagne_est_complete(tmp_path, monkeypatch):
    vec = tmp_path / "v" / rp.rb.MODELS["4b"].model_id / "joy_sadness.gguf"
    vec.parent.mkdir(parents=True)
    vec.write_bytes(b"x")
    out = tmp_path / "out"
    cfg = rp.Config(n_prompts=1)
    rp.check_manifest(out, rp.build_manifest(cfg, {"sha256": rp.rb.sha256(vec)}), False)
    steerer, _ = make_steerer()
    assert rp.run_campaign(steerer, prompts(1), out)
    import mood_steering.eval.tone as tone

    monkeypatch.setattr(tone.ToneScorer, "from_huggingface", classmethod(lambda c: FakeScorer()))
    report = tmp_path / "rapport.md"
    code = rp.main(
        ["--out", str(out), "--vectors-dir", str(tmp_path / "v"), "--n-prompts", "1",
         "--resume", "--report", str(report)]
    )  # fmt: skip
    assert code == 0
    assert report.read_text(encoding="utf-8").startswith("# H6")
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["h6"]["lecture"]
