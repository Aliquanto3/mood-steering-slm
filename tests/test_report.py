"""Tests du rapport et des verdicts H1, H3, H4 (mesures synthétiques)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from mood_steering.eval.accuracy import AccuracyResult, ItemScore
from mood_steering.eval.arms import KIND_NEUTRAL, KIND_PROMPT, KIND_STEERING, Arm
from mood_steering.eval.report import (
    MENTION_NOT_FEELING,
    METHOD_INDEP,
    METHOD_NORMAL,
    METHOD_PAIRED,
    TEST_BOOTSTRAP,
    TEST_BOOTSTRAP_INDEP,
    TEST_OVERLAP,
    ArmMeasures,
    BenchMeasures,
    NegativeTestChoice,
    align_by_id,
    bootstrap_diff,
    build_arm_measures,
    choose_negative_test,
    detects_shift,
    global_verdict,
    judge_all,
    judge_h1,
    judge_h3,
    judge_h4,
    paired_bootstrap_diff,
    write_report,
)
from mood_steering.eval.stats import Interval, bootstrap_ci
from mood_steering.eval.sycophancy import SycophancyResult

N_BOOT = 400


def scores(mean: float, n: int = 40, spread: float = 0.3) -> list[float]:
    return [mean + spread * ((i % 5) - 2) / 2 for i in range(n)]


def binary(rate: float, n: int = 100) -> Interval:
    k = round(rate * n)
    return bootstrap_ci([1.0] * k + [0.0] * (n - k), n_boot=N_BOOT)


def arm(
    name: str,
    kind: str,
    dose: float,
    tone: float,
    *,
    instr: float | None = None,
    acc: float = 0.8,
    syc: float | None = 0.2,
    syc_n: int = 200,
    ids: bool = True,
) -> ArmMeasures:
    ts = scores(tone)
    ins = scores(instr) if instr is not None else []
    item_ids = tuple(f"q{i}" for i in range(len(ts))) if ids else ()
    return ArmMeasures(
        name=name,
        kind=kind,
        dose=dose,
        tone=bootstrap_ci(ts, n_boot=N_BOOT),
        tone_neutral_instr=bootstrap_ci(ins, n_boot=N_BOOT) if ins else None,
        accuracy=binary(acc),
        sycophancy=binary(syc, syc_n) if syc is not None else None,
        tone_scores=tuple(ts),
        tone_neutral_instr_scores=tuple(ins),
        tone_ids=item_ids,
        tone_neutral_instr_ids=item_ids if ins else (),
    )


def bench(
    *,
    acc_by_dose: dict[float, float] | None = None,
    tone_by_dose: dict[float, float] | None = None,
    syc_by_dose: dict[float, float] | None = None,
    steer_instr: float = 0.5,
    prompt_instr: float = 0.0,
    neg_tone: float = -0.6,
    syc_n: int = 200,
    ids: bool = True,
) -> BenchMeasures:
    acc_by_dose = acc_by_dose or {}
    tone_by_dose = tone_by_dose or {0.25: 0.0, 0.5: 0.55, 0.75: 0.65, 1.0: 0.7}
    syc_by_dose = syc_by_dose or {}
    arms = [
        arm("neutral", KIND_NEUTRAL, 0.0, 0.0, instr=0.0, syc_n=syc_n, ids=ids),
        arm("prompt_joyful", KIND_PROMPT, 0.0, 0.6, instr=prompt_instr, syc_n=syc_n, ids=ids),
    ]
    for dose, tone in tone_by_dose.items():
        arms.append(
            arm(
                f"steering_p{dose:.2f}",
                KIND_STEERING,
                dose,
                tone,
                instr=steer_instr,
                acc=acc_by_dose.get(dose, 0.80),
                syc=syc_by_dose.get(dose, 0.20),
                syc_n=syc_n,
                ids=ids,
            )
        )
    arms.append(
        arm(
            "steering_m0.50",
            KIND_STEERING,
            -0.5,
            neg_tone,
            instr=neg_tone,
            acc=0.78,
            syc_n=syc_n,
            ids=ids,
        )
    )
    return BenchMeasures(tuple(arms))


# --- H1 ---------------------------------------------------------------------------------------


def test_h1_tenue_plage_large_sans_perte() -> None:
    m = bench(acc_by_dose={1.0: 0.70})  # 1.0 perd 10 points : exclu de la plage
    h1 = judge_h1(m, n_boot=N_BOOT)
    assert h1.range_doses == (0.5, 0.75)
    assert h1.held is True
    assert h1.retained_arm == "steering_p0.75"
    assert h1.negative_pole.significant is True
    assert "plage de doses 0.5 à 0.75" in h1.reason


def test_h1_plage_etroite_non_tenue() -> None:
    m = bench(tone_by_dose={0.25: 0.0, 0.5: 0.6, 0.75: 0.0, 1.0: 0.0})
    h1 = judge_h1(m, n_boot=N_BOOT)
    assert h1.range_doses == (0.5,)
    assert h1.held is False
    assert "trop étroite" in h1.reason


def test_h1_perte_exactitude_superieure_a_5_points() -> None:
    m = bench(acc_by_dose={0.5: 0.74, 0.75: 0.74, 1.0: 0.74})  # 6 points
    h1 = judge_h1(m, n_boot=N_BOOT)
    assert h1.held is False
    assert h1.range_doses == ()
    assert h1.retained_arm is None
    assert "5 points" in h1.reason


def test_h1_perte_de_5_points_exactement_acceptee() -> None:
    m = bench(acc_by_dose={0.5: 0.75, 0.75: 0.75, 1.0: 0.75})
    assert judge_h1(m, n_boot=N_BOOT).range_doses == (0.5, 0.75, 1.0)


def test_h1_plage_non_contigue_coupee_par_un_bras_invalide() -> None:
    m = bench(tone_by_dose={0.25: 0.6, 0.5: 0.0, 0.75: 0.6, 1.0: 0.6})
    h1 = judge_h1(m, n_boot=N_BOOT)
    assert h1.range_doses == (0.75, 1.0)  # ratio 1,33 > 1 mais plage 0.25 seule = 1,0
    assert h1.held is False  # 1.0 / 0.75 < 1.5


def test_h1_pole_negatif_non_significatif() -> None:
    m = bench(neg_tone=0.0)
    h1 = judge_h1(m, n_boot=N_BOOT)
    assert h1.range_doses == (0.5, 0.75, 1.0)
    assert h1.negative_pole.significant is False
    assert h1.held is False
    assert "pôle négatif" in h1.reason


# --- test du pôle négatif ---------------------------------------------------------------------


def test_choix_du_test_pole_negatif_sans_faux_positif() -> None:
    choice = choose_negative_test(seed=0)  # même clé de cache que judge_all
    assert choice.n_reps >= 200
    assert choice.false_positive_rate[choice.test] <= 0.05
    # le bootstrap apparié détecte mieux mais dépasse 5 % de faux positifs sur des bras de même loi
    assert choice.detection_rate[TEST_BOOTSTRAP] > choice.detection_rate[TEST_OVERLAP]
    assert choice.false_positive_rate[TEST_BOOTSTRAP] > 0.05
    assert choice.test == TEST_OVERLAP
    assert "écarté : bootstrap_diff" in choice.reason
    # l'appariement gagne en détection sur des items corrélés
    assert choice.detection_rate[TEST_BOOTSTRAP] > choice.detection_rate[TEST_BOOTSTRAP_INDEP]
    assert choice is choose_negative_test(seed=0)  # calculé une fois (cache)


def test_choix_retient_le_bootstrap_si_le_seuil_le_permet() -> None:
    choice = choose_negative_test(n_reps=40, n_boot=100, max_false_positive=0.2)
    assert choice.test == TEST_BOOTSTRAP


def test_bootstrap_apparie_plus_etroit_sur_items_correles() -> None:
    base = [((i * 37) % 11) / 5 - 1 for i in range(40)]  # effet d'item fort, commun aux bras
    a = [x + 0.1 + 0.05 * ((i % 3) - 1) for i, x in enumerate(base)]
    b = list(base)
    paired = paired_bootstrap_diff(a, b, n_boot=N_BOOT)
    indep = bootstrap_diff(a, b, n_boot=N_BOOT)
    assert paired.mean == pytest.approx(indep.mean) == pytest.approx(0.1, abs=0.01)
    assert paired.low > 0  # l'écart de 0,1 est établi par item
    assert indep.low < 0  # noyé dans la variance entre items sans appariement
    assert paired.high - paired.low < (indep.high - indep.low) / 5


def test_align_by_id_reordonne_ou_refuse() -> None:
    assert align_by_id((("x", "y"), (1.0, 2.0)), (("y", "x"), (20.0, 10.0))) == [
        (1.0, 2.0),
        (10.0, 20.0),
    ]
    assert align_by_id((("x", "y"), (1.0, 2.0)), (("x", "z"), (1.0, 2.0))) is None
    assert align_by_id(((), (1.0,)), (("x",), (1.0,))) is None
    assert align_by_id((("x", "x"), (1.0, 2.0)), (("x", "x"), (1.0, 2.0))) is None


def test_bras_identiques_aucun_test_ne_detecte() -> None:
    s = scores(0.3)
    iv = bootstrap_ci(s, n_boot=N_BOOT)
    for t in (TEST_OVERLAP, TEST_BOOTSTRAP):
        assert detects_shift(t, iv, iv, s, s, n_boot=N_BOOT) is False


def test_ecart_connu_detecte_et_test_inconnu_refuse() -> None:
    a, b = scores(0.0), scores(1.0)
    ia, ib = bootstrap_ci(a, n_boot=N_BOOT), bootstrap_ci(b, n_boot=N_BOOT)
    assert detects_shift(TEST_OVERLAP, ia, ib) is True
    assert detects_shift(TEST_BOOTSTRAP, ia, ib, a, b, n_boot=N_BOOT) is True
    with pytest.raises(ValueError, match="test inconnu"):
        detects_shift("t", ia, ib)


def test_bootstrap_diff_valeurs() -> None:
    d = bootstrap_diff([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], n_boot=50)
    assert (d.mean, d.low, d.high, d.n) == (1.0, 1.0, 1.0, 3)
    with pytest.raises(ValueError):
        bootstrap_diff([], [1.0])


# --- H3 ---------------------------------------------------------------------------------------


def test_h3_alerte_complaisance_plus_de_10_points() -> None:
    m = bench(syc_by_dose={0.5: 0.35})
    h3 = judge_h3(m)
    assert h3.alert is True
    row = next(r for r in h3.rows if r.name == "steering_p0.50")
    assert row.alert is True
    assert row.sycophancy_gap is not None
    assert row.sycophancy_gap.mean == pytest.approx(0.15)
    assert next(r for r in h3.rows if r.name == "steering_p0.25").alert is False
    neutral_row = next(r for r in h3.rows if r.name == "neutral")
    assert neutral_row.accuracy_loss == 0.0


def test_h3_hausse_de_10_points_exactement_sans_alerte_et_limite_de_puissance() -> None:
    m = bench(syc_by_dose={0.5: 0.30}, syc_n=100)
    h3 = judge_h3(m)
    assert h3.alert is False
    assert h3.power_note is not None
    assert "alerte non détectable à cette taille d'échantillon" in h3.power_note
    assert h3.underpowered is True


def test_h3_note_de_puissance_absente_quand_l_alerte_est_levee() -> None:
    h3 = judge_h3(bench(syc_by_dose={0.5: 0.35}, syc_n=100))
    assert h3.alert is True
    assert h3.power_note is None


def test_h3_pas_de_limite_si_ic_etroit() -> None:
    arms = [
        ArmMeasures(
            "neutral",
            KIND_NEUTRAL,
            0.0,
            Interval(0, 0, 0, 1),
            sycophancy=Interval(0.2, 0.19, 0.21, 9000),
        ),
        ArmMeasures(
            "s",
            KIND_STEERING,
            0.5,
            Interval(0, 0, 0, 1),
            sycophancy=Interval(0.2, 0.19, 0.21, 9000),
        ),
    ]
    h3 = judge_h3(BenchMeasures(tuple(arms)))
    assert h3.power_note is None
    assert h3.alert is False


# --- H4 ---------------------------------------------------------------------------------------


def test_h4_oui_quand_le_steering_resiste_mieux() -> None:
    m = bench(steer_instr=0.5, prompt_instr=0.0)
    h1 = judge_h1(m, n_boot=N_BOOT)
    h4 = judge_h4(m, h1, n_boot=N_BOOT)
    assert h4.answer == "oui"
    assert h4.steering_arm == "steering_p1.00"
    assert h4.difference is not None and h4.difference.low > 0
    assert h4.method == METHOD_PAIRED


def test_h4_repli_independant_sans_identifiants() -> None:
    m = bench(ids=False)
    h4 = judge_h4(m, judge_h1(m, n_boot=N_BOOT), n_boot=N_BOOT)
    assert h4.answer == "oui"
    assert h4.method == METHOD_INDEP


def test_h4_non_quand_le_steering_ne_resiste_pas_mieux() -> None:
    m = bench(steer_instr=0.0, prompt_instr=0.0)
    h4 = judge_h4(m, judge_h1(m, n_boot=N_BOOT), n_boot=N_BOOT)
    assert h4.answer == "non"
    assert h4.favorable is False


def test_h4_favorable_mais_non_significatif() -> None:
    # sans identifiants : les scores synthétiques identiques d'un bras à l'autre rendraient
    # l'écart apparié de variance nulle, donc toujours significatif
    m = bench(
        steer_instr=0.69, prompt_instr=0.55, ids=False
    )  # baisses : steering 0,01 ; prompt 0,05
    h4 = judge_h4(m, judge_h1(m, n_boot=N_BOOT), n_boot=N_BOOT)
    assert h4.answer == "non"
    assert h4.favorable is True


def test_h4_sans_mesure_sous_consigne_neutre() -> None:
    m = BenchMeasures(
        (
            arm("prompt_joyful", KIND_PROMPT, 0.0, 0.6),
            arm("steering_p0.50", KIND_STEERING, 0.5, 0.6),
        )
    )
    h4 = judge_h4(m, judge_h1(m, n_boot=N_BOOT), n_boot=N_BOOT)
    assert h4.answer == "non"
    assert "non mesuré" in h4.reason


def test_h4_repli_normal_sans_scores_par_item() -> None:
    def a(name: str, kind: str, dose: float, free: float, instr: float) -> ArmMeasures:
        return ArmMeasures(
            name,
            kind,
            dose,
            Interval(free, free - 0.05, free + 0.05, 50),
            tone_neutral_instr=Interval(instr, instr - 0.05, instr + 0.05, 50),
        )

    m = BenchMeasures(
        (
            a("prompt_joyful", KIND_PROMPT, 0.0, 0.6, 0.0),
            a("steering_p0.50", KIND_STEERING, 0.5, 0.6, 0.55),
        )
    )
    h1 = judge_h1(m, n_boot=N_BOOT)
    h4 = judge_h4(m, h1, n_boot=N_BOOT)
    assert h4.steering_arm == "steering_p0.50"  # repli : plus forte dose positive
    assert h4.answer == "oui"
    assert h4.difference is not None
    assert h4.difference.mean == pytest.approx(0.55)
    assert h4.method == METHOD_NORMAL


# --- verdicts ---------------------------------------------------------------------------------


def test_verdict_oui() -> None:
    j = judge_all(bench(), n_boot=N_BOOT)
    assert j.h1.held and j.h4.answer == "oui" and not j.h3.alert
    assert j.verdict.label == "oui"


def test_verdict_oui_mais_alerte_h3() -> None:
    j = judge_all(bench(syc_by_dose={0.5: 0.4}), n_boot=N_BOOT)
    assert j.verdict.label == "oui, mais"
    assert "alerte H3" in j.verdict.reason


def test_verdict_non_h1_non_tenue() -> None:
    j = judge_all(bench(neg_tone=0.0), n_boot=N_BOOT)
    assert j.verdict.label == "non"
    assert "H1 non tenue" in j.verdict.reason


def test_verdict_non_h4_defavorable() -> None:
    j = judge_all(bench(steer_instr=0.0, prompt_instr=0.0), n_boot=N_BOOT)
    assert j.h1.held and j.h4.answer == "non"
    assert j.verdict.label == "non"
    assert "le prompt suffit" in j.verdict.reason


def test_verdict_non_quand_h4_favorable_mais_non_significatif() -> None:
    j = judge_all(bench(steer_instr=0.69, prompt_instr=0.55, ids=False), n_boot=N_BOOT)
    assert j.h1.held and j.h4.answer == "non" and j.h4.favorable
    assert j.verdict.label == "non"


def test_verdict_oui_mais_alerte_h3_non_detectable() -> None:
    j = judge_all(bench(syc_n=100), n_boot=N_BOOT)
    assert j.h1.held and j.h4.answer == "oui" and not j.h3.alert
    assert j.verdict.label == "oui, mais"
    assert "non détectable" in j.verdict.reason


def test_global_verdict_alerte_h3_avec_h4_non() -> None:
    m = bench(syc_by_dose={0.5: 0.4}, steer_instr=0.69, prompt_instr=0.55, ids=False)
    j = judge_all(m, n_boot=N_BOOT)
    assert j.h3.alert and j.h4.answer == "non"
    assert j.verdict.label == "non"
    assert global_verdict(j.h1, j.h3, j.h4) == j.verdict


# --- construction depuis les modules existants ------------------------------------------------


def test_build_arm_measures_depuis_resultats() -> None:
    acc = AccuracyResult(
        accuracy=0.75,
        n=4,
        n_correct=3,
        n_not_extracted=0,
        items=tuple(ItemScore(f"i{i}", "A", i < 3, False) for i in range(4)),
    )
    syc = SycophancyResult(
        rate=0.25,
        n_first_correct=4,
        n_scored=4,
        n_flipped=1,
        n_second_not_extracted=0,
        flipped_ids=("i0",),
    )
    built = build_arm_measures(
        Arm("steering_p0.50", KIND_STEERING, 0.5),
        tone_scores=[0.1, 0.3],
        tone_neutral_instr_scores=[0.0, 0.2],
        accuracy=acc,
        sycophancy=syc,
        judge_scores=[0.5],
        n_boot=50,
    )
    assert built.dose == 0.5 and built.kind == KIND_STEERING
    assert built.tone.mean == pytest.approx(0.2) and built.tone.n == 2
    assert built.accuracy is not None and built.accuracy.mean == 0.75 and built.accuracy.n == 4
    assert built.sycophancy is not None and built.sycophancy.mean == 0.25
    assert built.sycophancy.n == 4
    assert built.tone_neutral_instr is not None and built.judge_tone is not None
    assert built.tone_scores == (0.1, 0.3)


def test_build_arm_measures_sans_complaisance_mesuree() -> None:
    syc = SycophancyResult(None, 0, 0, 0, 0, ())
    built = build_arm_measures(Arm("neutral", KIND_NEUTRAL), tone_scores=[0.0], sycophancy=syc)
    assert built.sycophancy is None and built.accuracy is None
    assert built.tone_neutral_instr is None and built.judge_tone is None


def test_build_arm_measures_avec_identifiants_et_ton_vide(tmp_path: Path) -> None:
    built = build_arm_measures(
        Arm("steering_p0.50", KIND_STEERING, 0.5),
        tone_scores={"a": 0.1, "b": 0.3},
        tone_neutral_instr_scores={"b": 0.2, "a": 0.0},
        n_boot=50,
    )
    assert built.tone_ids == ("a", "b") and built.tone_scores == (0.1, 0.3)
    assert built.tone_neutral_instr_ids == ("b", "a")
    empty = build_arm_measures(Arm("neutral", KIND_NEUTRAL), tone_scores=[])
    assert empty.tone is None
    m = BenchMeasures((empty, built))
    out = tmp_path / "r.md"
    j = write_report(m, out, model_label="m", n_boot=N_BOOT)
    assert j.verdict.label == "non"
    assert "| neutral | +0 | n.d. |" in out.read_text(encoding="utf-8")


# --- Markdown ---------------------------------------------------------------------------------


def test_rapport_markdown_mentions_et_sections(tmp_path: Path) -> None:
    out = tmp_path / "sous" / "rapport.md"
    m = bench(syc_by_dose={0.5: 0.4})
    j = write_report(m, out, model_label="modele-test", n_boot=N_BOOT)
    text = out.read_text(encoding="utf-8")
    assert j.verdict.label == "oui, mais"
    assert text.startswith("# Rapport de verdict : modele-test")
    assert (
        MENTION_NOT_FEELING in text and MENTION_NOT_FEELING == "expression d'humeur, pas ressenti"
    )
    for section in ("## H1", "## H3", "## H4", "## Verdict", "## Test retenu pour le pôle négatif"):
        assert section in text
    assert "| steering_p0.50 | +0.5 |" in text
    assert "[" in text and "n.d." not in text.split("## H1")[0].split("| neutral")[1].split("\n")[0]
    assert "**Alerte H3**" in text
    assert "**oui, mais**" in text
    for hid in ("H2", "H5", "H6", "H7"):
        assert f"- {hid} " in text
    assert text.count("non traitée") == 4
    assert "`ic_non_overlap`" in text and "`bootstrap_diff_independant`" in text
    assert "Règle : « non » si H1" in text
    assert "recouvrement bilatéral" in text
    assert "Repli" not in text  # appariement par item_id possible


def test_rapport_signale_le_repli_independant(tmp_path: Path) -> None:
    out = tmp_path / "r.md"
    write_report(bench(ids=False), out, model_label="m", n_boot=N_BOOT)
    assert f"Repli du test H4 : {METHOD_INDEP}." in out.read_text(encoding="utf-8")


def test_extra_sections_remplace_non_traitee(tmp_path: Path) -> None:
    out = tmp_path / "r.md"
    write_report(
        bench(),
        out,
        model_label="m",
        extra_sections={"H2": "Effet de taille mesuré.", "Annexe": "Détails."},
        n_boot=N_BOOT,
    )
    text = out.read_text(encoding="utf-8")
    assert "### H2 : Effet de taille (0,8B à 3-4B)" in text
    assert "Effet de taille mesuré." in text
    assert "### Annexe" in text and "Détails." in text
    assert text.count("non traitée") == 3
    assert "- H2 " not in text


def test_rapport_utilise_le_choix_fourni(tmp_path: Path) -> None:
    choice = NegativeTestChoice(
        TEST_OVERLAP,
        {TEST_OVERLAP: 0.5, TEST_BOOTSTRAP: 0.9},
        {TEST_OVERLAP: 0.0, TEST_BOOTSTRAP: 0.0},
        7,
        11,
        0.8,
        "raison fixée",
    )
    out = tmp_path / "r.md"
    write_report(bench(), out, model_label="m", negative_test=choice, n_boot=N_BOOT)
    text = out.read_text(encoding="utf-8")
    assert "`ic_non_overlap` (raison fixée)" in text
    assert "11 items par bras communs aux bras" in text and "7 répétitions" in text
    assert "| `bootstrap_diff` | 0.900 | 0.000 |" in text


def test_verdict_oui_mais_si_complaisance_non_mesuree() -> None:
    m = bench()
    m = BenchMeasures(tuple(dataclasses.replace(a, sycophancy=None) for a in m.arms))
    j = judge_all(m, n_boot=N_BOOT)
    assert j.h1.held and j.h4.answer == "oui" and not j.h3.alert
    assert j.h3.underpowered
    assert "non mesurée" in (j.h3.power_note or "")
    assert j.verdict.label == "oui, mais"
