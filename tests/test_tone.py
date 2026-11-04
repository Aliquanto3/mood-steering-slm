from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from mood_steering.eval import judge, stats, tone

EMOTION = {label: 0.0 for label in tone.EMOTION_LABELS}


def fake_emotion(texts):
    out = []
    for t in texts:
        p = dict(EMOTION)
        if "joie" in t:
            p.update(joy=0.8, sadness=0.05, neutral=0.15)
        elif "triste" in t:
            p.update(joy=0.1, sadness=0.7, neutral=0.2)
        else:
            p.update(neutral=1.0)
        out.append(p)
    return out


def fake_sentiment(texts):
    out = []
    for t in texts:
        p = {k: 0.0 for k in tone.SENTIMENT_LABELS}
        p["5 stars" if "joie" in t else "1 star" if "triste" in t else "3 stars"] = 1.0
        out.append(p)
    return out


def test_core_does_not_import_heavy_packages():
    code = (
        "import sys\n"
        "import mood_steering, mood_steering.eval.tone, mood_steering.eval.judge\n"
        "import mood_steering.eval.stats\n"
        "heavy = ('torch', 'transformers', 'anthropic', 'dotenv')\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_joy_minus_sadness_scores():
    scorer = tone.ToneScorer(fake_emotion, fake_sentiment)
    scores = scorer.score(["de la joie", "c'est triste", "table"])
    assert scores == pytest.approx([0.75, -0.6, 0.0])


def test_control_polarity_formula():
    scorer = tone.ToneScorer(fake_emotion, fake_sentiment)
    assert scorer.control(["de la joie", "c'est triste", "table"]) == pytest.approx([1, -1, 0])
    mixed = {"1 star": 0.5, "2 stars": 0, "3 stars": 0, "4 stars": 0, "5 stars": 0.5}
    assert tone.stars_polarity(mixed) == 0.0
    assert (
        tone.stars_polarity({"1 star": 0, "2 stars": 0, "3 stars": 0, "4 stars": 1, "5 stars": 0})
        == 0.5
    )


def test_missing_labels_raise():
    with pytest.raises(ValueError, match="sadness"):
        tone.joy_minus_sadness({"joy": 0.5})
    with pytest.raises(ValueError, match="5 stars"):
        tone.stars_polarity({"1 star": 1.0, "2 stars": 0, "3 stars": 0, "4 stars": 0})
    scorer = tone.ToneScorer(lambda t: [{"joy": 1.0}] * len(t))
    with pytest.raises(ValueError, match="libellés"):
        scorer.score(["a"])


def test_control_without_classifier_and_length_mismatch():
    with pytest.raises(ValueError, match="contrôle"):
        tone.ToneScorer(fake_emotion).control(["a"])
    with pytest.raises(ValueError, match="résultats"):
        tone.ToneScorer(lambda t: []).score(["a"])


def records():
    return [
        {"arm": arm, "item_id": f"i{k}", "output": f"sortie {arm} {k}"}
        for arm in ("neutre", "prompt", "steer+1")
        for k in range(10)
    ]


def test_batch_masks_arm_and_is_deterministic():
    b1 = judge.prepare_judge_batch(records(), 12, seed=3)
    b2 = judge.prepare_judge_batch(records(), 12, seed=3)
    b3 = judge.prepare_judge_batch(records(), 12, seed=4)
    assert b1 == b2
    assert [i.blind_id for i in b1.items] != [i.blind_id for i in b3.items]
    assert len(b1.items) == 12 and len(b1.mapping) == 12
    for item in b1.items:
        assert item.blind_id.startswith("j")
        assert "neutre" not in item.blind_id and "prompt" not in item.blind_id
        arm, item_id = b1.mapping[item.blind_id]
        assert item.text == f"sortie {arm} {item_id[1:]}"
    # l'ordre est mélangé : pas groupé par bras comme l'entrée
    arms = [b1.mapping[i.blind_id][0] for i in b1.items]
    assert arms != sorted(arms)


def test_batch_sample_larger_than_records_keeps_all():
    batch = judge.prepare_judge_batch(records(), 1000, seed=0)
    assert len(batch.items) == 30
    with pytest.raises(ValueError):
        judge.prepare_judge_batch(records(), 0, seed=0)


def test_estimate_cost_exact():
    batch = judge.JudgeBatch((judge.JudgeItem("j1", "a" * 300),), {"j1": ("x", "1")})
    # 100 + 120 tokens en entrée à 4 $/Mtok, 25 en sortie à 20 $/Mtok
    assert judge.estimate_cost(batch) == pytest.approx((220 * 4 + 25 * 20) / 1e6)
    with pytest.raises(ValueError, match="prix inconnu"):
        judge.estimate_cost(batch, model="autre")


def test_default_sample_cost_under_cap():
    long = [{"arm": "a", "item_id": str(i), "output": "x" * 400} for i in range(300)]
    batch = judge.prepare_judge_batch(long, judge.DEFAULT_SAMPLE_SIZE, seed=0)
    cost = judge.estimate_cost(batch)
    assert 0.25 < cost < 0.40 < judge.DEFAULT_MAX_COST_USD


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(text=self.replies.pop(0))])


def test_run_judge_refuses_over_cap_without_calling():
    batch = judge.prepare_judge_batch(records(), 5, seed=0)
    client = FakeClient([])
    with pytest.raises(RuntimeError, match="plafond"):
        judge.run_judge(batch, client, max_cost_usd=1e-6)
    assert client.calls == []


def test_run_judge_scores_and_attaches_to_arm():
    batch = judge.prepare_judge_batch(records(), 4, seed=1)
    replies = ['{"score": 0.5}', '```json\n{"score": -1}\n```', "pas de json", '{"score": 3}']
    client = FakeClient(replies)
    result = judge.run_judge(batch, client)
    ids = [i.blind_id for i in batch.items]
    assert result.scores == {ids[0]: 0.5, ids[1]: -1.0}
    assert set(result.failures) == {ids[2], ids[3]}
    # le juge ne voit ni le bras ni l'identifiant d'item
    for call, item in zip(client.calls, batch.items, strict=True):
        assert call["model"] == "claude-opus-5-5"
        assert item.text in call["messages"][0]["content"]
        arm, item_id = batch.mapping[item.blind_id]
        assert arm not in call["messages"][0]["content"].replace(item.text, "")
    by_arm = judge.scores_by_arm(batch, result.scores)
    assert sorted(v for vs in by_arm.values() for v in vs) == [-1.0, 0.5]
    assert by_arm[batch.mapping[ids[0]][0]].count(0.5) == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"score": 0.25}', 0.25),
        ('Voici : {"score": -0.5, "raison": "x"} fin', -0.5),
        ('{"score": 1}', 1.0),
    ],
)
def test_parse_valid(text, expected):
    assert judge.parse_judge_response(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "bonjour",
        "{pas json}",
        '{"autre": 1}',
        '{"score": "0.5"}',
        '{"score": 2}',
        '{"score": true}',
        '{"score": NaN}',
        "[0.5]",
    ],
)
def test_parse_invalid(text):
    with pytest.raises(judge.JudgeParseError):
        judge.parse_judge_response(text)


def test_resolve_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert judge.resolve_api_key(tmp_path / "absent.env") == "from-env"
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="introuvable"):
        judge.resolve_api_key(tmp_path / "absent.env")


def test_find_env_file_order(tmp_path, monkeypatch):
    monkeypatch.delenv(judge.ENV_FILE_VAR, raising=False)
    repo = tmp_path / "repo"
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    # pas de .env : rien
    assert judge.find_env_file(sub) is None
    # .env à la racine du dépôt, trouvé en remontant
    (repo / ".env").write_text("ANTHROPIC_API_KEY=racine\n", encoding="utf-8")
    assert judge.find_env_file(sub) == repo / ".env"
    # .env du répertoire courant prioritaire
    (sub / ".env").write_text("ANTHROPIC_API_KEY=courant\n", encoding="utf-8")
    assert judge.find_env_file(sub) == sub / ".env"
    # variable explicite prioritaire sur tout
    explicit = tmp_path / "ailleurs.env"
    monkeypatch.setenv(judge.ENV_FILE_VAR, str(explicit))
    assert judge.find_env_file(sub) == explicit


def test_resolve_api_key_from_env_file(tmp_path, monkeypatch):
    pytest.importorskip("dotenv")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(judge.ENV_FILE_VAR, raising=False)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=depuis-fichier\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert judge.resolve_api_key() == "depuis-fichier"


def test_run_judge_api_error_keeps_other_scores():
    batch = judge.prepare_judge_batch(records(), 3, seed=2)

    class Flaky(FakeClient):
        def create(self, **kw):
            self.calls.append(kw)
            if len(self.calls) == 2:
                raise ConnectionError("secret sk-ant-xyz")
            return SimpleNamespace(content=[SimpleNamespace(text='{"score": 0.1}')])

    client = Flaky([])
    result = judge.run_judge(batch, client)
    ids = [i.blind_id for i in batch.items]
    assert len(client.calls) == 3
    assert result.scores == {ids[0]: 0.1, ids[2]: 0.1}
    assert result.failures == {ids[1]: "erreur API : ConnectionError"}


def test_bootstrap_deterministic_and_contains_mean():
    values = [0.1, 0.4, -0.2, 0.9, 0.3, 0.0, 0.5, -0.1]
    a = stats.bootstrap_ci(values, n_boot=500, seed=7)
    b = stats.bootstrap_ci(values, n_boot=500, seed=7)
    assert a == b
    assert a.mean == pytest.approx(sum(values) / len(values))
    assert a.low <= a.mean <= a.high
    assert a.low < a.high and a.n == 8
    assert a != stats.bootstrap_ci(values, n_boot=500, seed=8)


def test_bootstrap_constant_values_and_errors():
    ci = stats.bootstrap_ci([0.5, 0.5, 0.5], n_boot=50)
    assert (ci.mean, ci.low, ci.high) == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])
    with pytest.raises(ValueError):
        stats.bootstrap_ci([1.0], n_boot=0)


def test_aggregate_per_arm():
    res = stats.aggregate({"a": [1.0, 1.0], "b": [-1.0, 0.0, 1.0]}, n_boot=100, seed=1)
    assert set(res) == {"a", "b"}
    assert res["a"].mean == 1.0 and res["b"].mean == 0.0
    assert res["b"].low <= 0.0 <= res["b"].high
