---
title: 'A4 : mesure H7 répétée, ordre des bras alterné'
type: feature
created: '2026-11-05'
status: review
context:
  - _bmad-output/implementation-artifacts/epic-5-retro-2026-11-05.md
---

## Intention

**Problème.** `run_bench.py --energy` prend une seule lecture CodeCarbon par bras, bras mesurés dans un ordre fixe : l'ordre se confond avec la dérive de charge et l'énergie n'a pas d'intervalle (rétro de l'epic 5, F14, action A4).

**Approche.** `--energy-repeats K` : K tours de mesure de la phase ton, ordre des bras alterné, lectures stockées par tour, rapport avec moyenne et IC95 sur les tours et ratio apparié par tour.

## Décisions

- Règle d'alternance : tour pair dans l'ordre des bras, tour impair dans l'ordre inverse (`ENERGY_ROUND_ORDER`, figée au manifeste avec K, absents à K = 1).
- Tour 0 = la génération notée habituelle ; tours 1 à K-1 dans `{modèle}/tone_r{r}/{bras}.jsonl` (hors des phases, jamais notés), lancés après la phase ton du modèle. Mêmes prompts, mêmes graines.
- Stockage : `energy_units.json` garde ses clés ; tour r > 0 sous `{modèle}/tone#r{r}/{bras}`. Fichiers K = 1 relus à l'identique (`data/runs/energie-5-3` : section H7 régénérée identique octet pour octet).
- Reprise : unité par (tour, bras), `.part` repris à l'item, lecture cumulée comme au tour 0 ; une lecture sans `stop()` (arrêt brutal) rend le tour non publié. Autre K refusé (`check_runtime` et manifeste). Même garde du `finally` que A5 (`_measured`).
- `--tone-only` (ajouté, nécessaire) : phase ton seule pour le cœur aussi, car `--short-run` refuse le 4B. Sans QCM ni notation ; rapport de frugalité seul (`write_frugality_report`, extrait de `build_reports`). Clés de manifeste absentes hors option.
- Relevé de charge au lancement (`frugality.system_load` : `nvidia-smi`, CPU via `psutil` si présent, `None` sinon) dans `runs.jsonl`, pas au manifeste (il varie d'une reprise à l'autre et ferait refuser la reprise).
- Pas de verdict H7 : seuils non pré-enregistrés (A6).

## Reports

- Mesure 4B elle-même (GPU occupé par une autre mesure) : commande et durée estimée dans `docs/rapport-frugalite.md`, section méthode.
- `docs/rapport-frugalite.md` non régénéré (K = 1, texte inchangé).
