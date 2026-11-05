# Story A5 — Mise en commun v0.3 (JSONL reprenable, bootstrap apparié, critère de dose utile, chemin d'énergie mort)

Source : rétro de l'epic 5 (`epic-5-retro-2026-11-05.md`, constats F3, F4, F5, action A5). Accord d'Anaël du 2026-11-05 pour le développement et l'exécution.

## Contexte

- Quatre stratégies de reprise JSONL coexistent : `scripts/run_bench.py` (`_read_part`), `run_pulse.py`, `run_antiloop.py` (passe par `rb._read_part`, fonction privée), `run_axis.py`.
- Deux critères de « ton déplacé » : IC séparés disjoints dans `report_size.py:164-173` (5.2) ; bootstrap apparié de la différence dans `run_pulse.py:276-303` (5.1) et `run_axis.py:365-376` (5.4). Une même dose peut être « non utile » dans `rapport-taille.md` et significative dans les rapports H5/H6.
- Deux générateurs aléatoires et deux copies du percentile (`eval/stats.py:19`, `report_size.py:258`).
- `run_axis` charge `run_bench` par `importlib` sous `sys.modules["run_bench"]`, `report_size` par `sys.path`.
- `runner.run_bench(energy_tracker=…)` (`eval/runner.py:118-170`) n'est appelé par aucun script ; la campagne passe par `run_bench.py` et `energy_units.json`.

## Décisions

1. Nouveau module `src/mood_steering/eval/resumable.py` : lecture JSONL découpée sur `\n` (jamais `splitlines()`, cf. U+2028), dernière ligne incomplète ignorée, réécriture atomique (`.part` + `os.replace`). Les quatre scripts l'utilisent ; plus aucune lecture/reprise propre à un script.
2. `src/mood_steering/eval/stats.py` porte seul `percentile`, l'IC par bootstrap et `paired_bootstrap_diff(a, b, *, n_boot, seed)` avec un seul générateur (`random.Random(seed)`). Les copies de `report_size`, `run_pulse`, `run_axis` sont supprimées.
3. Critère unique de « ton déplacé » / dose utile : **IC95 du bootstrap apparié de la différence (bras − neutre) qui exclut 0**. `report_size` l'adopte.
4. Aucun script n'importe une fonction privée d'un autre script ni ne manipule `sys.modules` ; ce qui est partagé vit dans le paquet.
5. Chemin d'énergie mort retiré : paramètre `energy_tracker` de `runner.run_bench` supprimé ; les tests de `test_frugality.py` qui le couvraient visent désormais le chemin réel (`energy_units.json` de `run_bench.py`).

## Critères d'acceptation

- **Given** un JSONL contenant un U+2028 dans une sortie et une dernière ligne tronquée, **When** un des quatre scripts reprend un run, **Then** aucune ligne valide n'est perdue et la ligne tronquée est ignorée (un test par script ou un test du module plus un test d'usage par script).
- **Given** les mêmes données et la même graine, **When** on calcule la dose utile dans `report_size`, `run_pulse` et `run_axis`, **Then** les trois appellent la même fonction de `eval/stats.py` et un test le vérifie sur des données synthétiques.
- **Given** le dépôt, **When** on cherche `_read_part`, `_percentile`, `_pct`, `importlib`, `energy_tracker`, **Then** il ne reste aucun doublon ni chemin mort.
- **Given** les données existantes de `data/runs/` (non versionnées), **When** on régénère les rapports sans génération (`--report-only` ou équivalent), **Then** `rapport-taille.md`, `rapport-pharmacocinetique.md` et `rapport-axe-calme.md` sont recalculés ; tout chiffre ou verdict qui change est signalé, et `rapport-verdict.md` est réaligné.
- `uv run pytest -q --tb=short`, `uv run ruff check .`, `uv run ruff format --check .` verts.
