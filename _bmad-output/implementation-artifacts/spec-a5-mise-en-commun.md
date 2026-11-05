---
title: 'A5 : mise en commun v0.3'
type: refactor
created: '2026-11-05'
status: review
context:
  - _bmad-output/implementation-artifacts/epic-5-retro-2026-11-05.md
---

## Intention

**Problème.** Quatre reprises JSONL, deux critères de « ton déplacé », deux générateurs aléatoires, un chargement de script par `importlib` et un chemin d'énergie jamais appelé coexistaient (rétro de l'epic 5, F3 à F5, action A5).

**Approche.** Mutualiser dans le paquet : `eval/resumable.py` (lecture sur `\n` seul, dernière ligne tronquée ignorée, réécriture atomique) et `eval/stats.py` (un percentile, un générateur `random.Random(seed)`, `paired_bootstrap_diff`, `paired_bootstrap_stat`, `excludes_zero`, `paired_shift`).

## Décisions

- Critère unique de dose utile : IC95 du bootstrap apparié de (bras − neutre) qui exclut 0 du côté de la dose. `report_size` l'adopte ; `run_pulse` (`positive`) et `run_axis` (`Ci.excludes_zero`) passent par `excludes_zero`.
- `run_axis` importe `run_bench` par `sys.path`, plus de `sys.modules` ni de `importlib` ; `rb._read_part` et `rb._git_sha` (désormais `git_sha`) ne sont plus privés.
- `runner.run_bench(energy_tracker=…)` retiré ; le chemin réel est `run_bench.py --energy` (`energy_units.json`), couvert par `test_run_bench.py`.
- `run_pulse.py --report-only` : rapport depuis les sorties et les tons de `summary.json`, sans génération ni classifieur (`run_axis.py` et `report_size.py` en avaient déjà un).
- Formats inchangés (JSONL, `config.json`, `energy_units.json`, `summary.json`).

## Résultat de la régénération (données de `data/runs/`, sans génération)

- `rapport-taille.md` : aucun chiffre ni verdict ne bouge (les 18 bras steering restent « ton déplacé » aux deux critères) ; seul le texte de méthode change.
- `rapport-pharmacocinetique.md` : identique (le critère H6 était déjà apparié).
- `rapport-axe-calme.md` : points inchangés ; bornes d'IC95 décalées de quelques centièmes (générateur numpy → `random.Random`) ; verdicts H5 inchangés ; bornes citées dans la lecture réalignées.

## Reports

- Le `finally` de `run_bench.run_campaign` ne protège pas l'exception du bras si `energy_tracker.stop()` échoue (le chemin mort le faisait) : à traiter à part.
