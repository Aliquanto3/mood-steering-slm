# Story A4 — Mesure H7 répétée, ordre des bras alterné

Source : rétro de l'epic 5, constat F14 et action A4 (« mesurer H7 sur le 4B en GPU, machine au repos, avec plusieurs lectures par bras et l'ordre des bras alterné »). Lancée en autonomie le 2026-11-05, GPU laissé à l'orchestrateur pour la nuit.

## Contexte

`scripts/run_bench.py --energy` (chemin réel depuis A5) prend une seule lecture CodeCarbon par unité (modèle × bras de la phase ton), bras mesurés l'un après l'autre dans un ordre fixe : l'ordre se confond avec la dérive de charge et il n'y a pas d'intervalle sur l'énergie (`docs/rapport-frugalite.md`).

## Décisions

1. Nouvelle option `--energy-repeats K` (défaut 1, comportement actuel inchangé). Avec K > 1, la phase ton est mesurée en K tours ; chaque tour génère les prompts de ton de chaque bras et prend une lecture CodeCarbon par bras et par tour.
2. Ordre alterné : l'ordre des bras change à chaque tour (tour pair dans l'ordre, tour impair dans l'ordre inverse, ou rotation ; le choix est documenté et figé dans le manifeste).
3. Les sorties de génération restent celles d'un seul passage (le premier tour) pour la notation du ton ; les tours suivants ne servent qu'à l'énergie et à la latence. Aucune double notation.
4. `energy_units.json` reste lisible pour les runs existants (K = 1) ; les lectures répétées sont stockées par tour, reprise d'un tour interrompu comprise, et une reprise sous un autre K est refusée comme pour `--energy`.
5. Rapport H7 : par bras, énergie et CO2 par génération avec moyenne et IC95 sur les tours (bootstrap de `eval/stats.py`), latence comme aujourd'hui, et ratio steering / `prompt_joyful` avec IC apparié par tour. Le manifeste note K, l'ordre des tours, le device, et un relevé de charge au début (utilisation GPU via `nvidia-smi` si présent, charge CPU), pour documenter « machine au repos ».
6. Pas de verdict H7 : les seuils des hypothèses bonus ne sont pas encore pré-enregistrés (action A6, décision d'Anaël). Le rapport donne les mesures et leurs intervalles, et le dit.

## Critères d'acceptation

- **Given** `--energy --energy-repeats 3` sur un modèle simulé à 3 bras, **When** la campagne tourne, **Then** chaque bras a 3 lectures, l'ordre des bras diffère d'un tour à l'autre selon la règle documentée, et les sorties notées sont celles d'un seul passage.
- **Given** une campagne interrompue au milieu du tour 2, **When** on la reprend avec `--resume`, **Then** les lectures des tours finis sont conservées, le tour 2 reprend sans doublon, et une reprise avec un autre K est refusée.
- **Given** un `energy_units.json` d'un run existant (K = 1, ex. `data/runs/energie-5-3`), **When** on régénère le rapport, **Then** il est relu sans erreur et le rapport est inchangé hors mention de K.
- **Given** K lectures par bras, **When** le rapport H7 est produit, **Then** il donne moyenne et IC95 par bras et un ratio apparié par tour face à `prompt_joyful`, sans verdict.
- Tests sans modèle réel ni CodeCarbon réel (traqueur simulé) ; `uv run pytest -q --tb=short`, `uv run ruff check .`, `uv run ruff format --check .` verts.
