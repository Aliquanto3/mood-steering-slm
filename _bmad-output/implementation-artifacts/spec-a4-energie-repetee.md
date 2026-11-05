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

## Revue

### Passe 1 (2026-11-05, SHA revu d4dcb12)

Branche à jour d'origin/main (b3bf8ef). Rapport K = 1 de `data/runs/energie-5-3` régénéré : section H7 identique à `docs/rapport-frugalite.md`.

- Important, medium, second avis 1 : IC95 bootstrap percentile sur K tours sous-couvrant (0,74 à K = 3). Corrigé : `stats.student_ci` (t à n - 1 ddl, table jusqu'à 30, Cornish-Fisher au-delà ; couverture simulée 0,95 à n = 3 et 6), pour l'énergie par bras et le ratio apparié ; bootstrap des autres usages inchangé.
- Important, medium, second avis 2 : tour 0 non comparable (juste après chargement, ordre fixe). Corrigé : tour 0 = sorties notées et chauffe, hors statistiques ; alternance à partir du tour 1 (impairs dans l'ordre, pairs inversés) ; K pair refusé (`--energy-repeats` impair) ; avertissement au rapport si le nombre de tours mesurés est impair ; K = 1 inchangé. Commande documentée : `--energy-repeats 7`.
- Important, medium, second avis 3 : `compute_minutes` comptait `tone_r*`. Corrigé (exclus).
- Important, medium, second avis 4 : tour commencé sans lecture (arrêt brutal) publié sans énergie, en silence. Corrigé : à la reprise, `.part` sans lecture supprimé et tour remesuré en entier (tour non noté) ; tout couple (tour, bras) sans lecture est listé au rapport.
- Nit, low, second avis 5 : `--tone-only` sans `--report` réécrivait `docs/rapport-frugalite.md`. Corrigé : défaut `{out}/frugalite.md`, `docs/rapport-verdict.md` refusé.
- Nit, low, second avis 6 : « 8 bras x 5 tours » faux pour le 4B. Corrigé : 5 bras x 7 tours = 35 passes, 1,75 à 3 h.
- Nit, low, second avis 7 : `energy_units.json` non atomique. Corrigé (`.part` + `os.replace`).
- Nit, low, second avis 8 : `system_load` ne captait que `ImportError` côté CPU. Corrigé (toute exception, relevé jamais bloquant).
- Nit, low, second avis 9 : `--tone-only` lève le plancher QCM sur le 4B. Pas un bug (aucun QCM) ; documenté dans la méthode.
- Pre-existing, low, revue : reprise d'une unité déjà lue partiellement puis tuée avant `stop()` : la seconde portion n'a pas de lecture (sous-comptage). Même mécanisme qu'au tour 0 depuis A5 ; non corrigé ici.
- Complexité : 0 constat (lectures `read_jsonl` par tour et bras au seul rapport ; `energy_units.json` réécrit par unité, quelques dizaines de clés).

### Passe 2 (2026-11-05, SHA revu 1d4d438)

Branche à jour d'origin/main. Relu `d4dcb12..1d4d438` ; 519 tests passent, ruff propre.

- Aucun constat Important. Table `_T975` vérifiée valeur par valeur (ddl 1 à 30) ; `student_ci` : n = 2 (t = 12,706, IC large), n = 1 (nan, affiché « IC n/d »), n = 0 (erreur) ; Cornish-Fisher au-delà de 30 (2,0394 à 31, 1,9623 à 1000) exact.
- Suppression du `.part` : limitée à `_run_rounds` (tours r >= 1, chemin `tone_r{n}`) et seulement si aucune lecture n'existe ; le tour 0 noté et les tours déjà mesurés (`path.exists()` ou lecture présente, donc reprise cumulative) ne sont pas touchés.
- K = 1 : `repeat_units` vide, aucune clé `energy_repeats`, ordre inchangé. Refus de K pair : seul `--energy-repeats > 1` est visé, le défaut 1 reste valable ; aucune campagne K pair publiée (fonction non fusionnée).
- Pre-existing/Nit : aucun.
