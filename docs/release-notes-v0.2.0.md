# mood-steering-slm v0.2.0

Deuxième lot : les hypothèses bonus (epic 5) et l'ablation de direction à l'inférence (epic 6). Le démonstrateur montre l'expression d'une humeur dans le texte, jamais un ressenti.

## Livré

- **5.1, pharmacocinétique (H6).** `scripts/run_pulse.py`, profil `pulse` dans la démo. Bras de contrôle `constant_gen` et `decay` pour séparer l'effet sur le prompt de l'effet pendant la génération. Rapport `docs/rapport-pharmacocinetique.md`.
- **5.2, effet de taille (H2).** `scripts/report_size.py`, rapport `docs/rapport-taille.md`.
- **5.3, frugalité (H7).** Options `--energy`, `--energy-country` et `--short-run` de `run_bench.py` (CodeCarbon), rapport `docs/rapport-frugalite.md`.
- **5.4, second axe calme ↔ agitation (H5).** 107 paires contrastées, `scripts/run_axis.py` (cosinus entre axes, bras combiné, verdict, `--report-only`), rapport `docs/rapport-axe-calme.md`.
- **6.1 et 6.2, ablation de direction et mesure anti-boucle.** `DirectionAblator` (PyTorch, extra `ablation`), `scripts/run_antiloop.py` (origine, chirurgie des poids, ablation à deux doses), rapport `docs/rapport-anti-boucle.md`.

## Résultats

- H6 : non pour le pulse. Oui, sous une forme réduite, pour la décroissance seule (+0,41 [+0,29 ; +0,53]). La dose posée sur le prompt agit vite et fort, celle posée pendant la génération agit plus faiblement et plus tard.
- H2 : le 4B produit moins de sorties dégénérées à +1 et un ton plus marqué que le 2B et le 0.8B, mais les doses sont relatives au `max_scale` de chaque modèle : aucun effet de taille n'est établi.
- H7 : mesure courte sur le 0.8B en CPU, sans écart d'énergie détectable. L'écart de latence (0,05 à 0,14 s en faveur du steering) n'est pas établi.
- H5 : non concluante. L'axe se distingue de joie ↔ tristesse (cosinus de +0,28 à +0,36), mais seule l'échelle 0,2 donne des sorties valides, le calme baisse aussi la valence et les proxies d'activation ne sont pas validés.
- Anti-boucle : non concluant. Avec un plafond de 512 tokens, 96 à 97 % des réponses sont tronquées dans les quatre bras, dont les IC se recouvrent.
- Rejugement en aveugle de la campagne 3.5 à 128 tokens : corrélation de 0,82 avec le classifieur. Synthèse : [`rapport-verdict.md`](rapport-verdict.md).

## Limites et suites

- Mesures de 5.4 sur CPU, aux échelles 0,2 et 0,4 seulement. Il reste à calibrer le calme, mesurer 0,3 et 0,5 sur GPU, équilibrer la valence des paires puis régénérer le vecteur, et trouver un proxy d'activation validé.
- Anti-boucle : relancer avec un plafond de 2048 ou 4096 tokens sur un n réduit, et faire le contrôle croisé avec slm-quality-bench.
- Pistes retenues pour une v0.3 : un plafond de dose calculé axe par axe d'après le critère de fenêtre de contrôle (arXiv 2606.19831) ; un steering restreint aux neurones les plus discriminants des paires contrastées (arXiv 2605.12290, 2601.22594) ; une démo dans le navigateur en WebGPU.
- Poids, vecteurs et données ne sont pas dans le dépôt.

## Étapes humaines restantes

- Relecture des 107 paires calme ↔ agitation (`src/mood_steering/pairs/calm_agitation.jsonl`).
- Test de `docs/demo.md` sur le portable, pour confirmer la vitesse réelle.

Aucune ne bloque la release.
