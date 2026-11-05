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

- Aucun (le `finally` de `run_campaign` est corrigé en revue, passe 1).

## Revue

### Passe 1 (2026-11-05, SHA revu 3890be9)

- Important · medium · second avis · corrigé : `run_campaign` laissait une erreur de `energy_tracker.stop()` masquer l'exception de l'unité (garde perdue avec le chemin mort de `runner`). Lecture partielle sauvée si possible, échec du traqueur journalisé, exception de l'unité relancée ; un `stop()` en échec après une unité réussie lève. Tests `test_stop_failure_does_not_mask_unit_exception` et `test_stop_failure_after_successful_unit_is_raised` sur le chemin réel.
- Important · low · second avis · corrigé : `run_bench.read_jsonl` (fichiers terminés, `runs.jsonl`) tronquait en silence à une ligne illisible ; une ligne collée par un ajout après arrêt brutal dans `runs.jsonl` aurait fait sauter le contrôle d'environnement des reprises suivantes. `read_rows(strict=True)` lève `ValueError` ; vérifié sans erreur sur les 142 JSONL de `data/runs/`.
- Nit · low · second avis · corrigé : `sycophancy.read_jsonl` coupait par `splitlines()` (U+2028) ; passe par `read_rows(strict=True)`.
- Nit · low · second avis · rejeté : `Ci.excludes_zero` appelé par les seuls tests ; les verdicts de `run_axis` sont directionnels (`high < 0`, `low > 0`), équivalents à `excludes_zero(iv, ±1)` nan compris (comparaison fausse).
- Nit · low · second avis · rejeté : test U+2028 + ligne tronquée pour `report_size` et `run_axis` ; les deux lisent par `resumable`, couvert dans `test_resumable.py`.
- Nit · low · second avis · rejeté : tirages nan écartés dans `paired_boot` ; les colonnes de `run_axis` sont filtrées des nan avant le bootstrap (moyennes), aucun tirage nan possible.
- Nit · low · second avis · accepté : lecture arrêtée à la première ligne illisible en reprise, comportement antérieur, documenté dans `resumable`.
- Complexité : 0 constat (lectures complètes une fois par reprise ou par rapport).

### Passe 2 (2026-11-05, SHA revu 6966491)

- Aucun Important. Correctif du `finally` : `except BaseException` conserve la lecture partielle, journalise l'échec du traqueur et relance l'exception de l'unité ; sur succès, l'échec de `stop()` lève. Les deux tests passent par `run_campaign` réel.
- Nit · low · reporté : `runs.jsonl` est écrit en fin de campagne (une ligne), seule lecture stricte exposée à une ligne tronquée par un arrêt pendant cette écriture ; `ValueError` explicite, correction manuelle. Les unités terminées sont écrites via `.part` puis `os.replace` : aucune ligne tronquée légitime n'atteint `strict`. La reprise (`run_unit`, `run_pulse`) reste sur `read_rows` tolérant.
- Complexité : 0 constat.
