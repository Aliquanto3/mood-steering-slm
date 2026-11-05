# Rapport anti-boucle : chirurgie des poids ou ablation dosable

Statut : mesure complète faite le 2026-11-05 (story bonus 6.2, SPEC CAP-9), run `antiloop-full`. Résultat : le plafond de 512 tokens sature tous les bras, la mesure ne discrimine pas (voir Lecture et limites).

## Question

Sur Qwen3.5-0.8B, le MIT retire une direction (couches 15 et 19) des poids pour faire cesser les boucles de raisonnement ([slm-weight-editing](https://github.com/Aliquanto3/slm-weight-editing)). La même direction, ablatée à l'inférence avec une dose `alpha`, donne-t-elle un effet comparable sur le taux de réponses tronquées, sans modifier les poids ? Le rapport compare la chirurgie (permanente) à la molécule (dosable, réversible).

## Méthode

Trois bras, même protocole : mêmes items et même ordre, génération gloutonne, même plafond `max_new_tokens` (512 par défaut), même gabarit de chat (`enable_thinking=True` : le `<think>` est ouvert, car les boucles sont des boucles de raisonnement ; le gabarit Qwen3.5 le referme sinon et la sortie se limite à la lettre). Pile : transformers.

| Bras | Modèle | Intervention |
|---|---|---|
| `origin` | `qwen3.5-base` | aucune |
| `surgery` | `qwen3.5-wsp` | poids édités (MIT) |
| `ablation@<alpha>` | `qwen3.5-base` | `DirectionAblator`, une direction par couche (15 et 19), sortie de la couche, doses 0,5 et 1,0 |

Les directions viennent du diff du MIT (fichiers `layer15.npy` et `layer19.npy`, chargés par `scripts/run_antiloop.py`) ; elles ne sont pas versionnées.

Jeu : `load_global_mmlu_lite(n=100)`, QCM en français, graine fixe. Mesures par bras :

- taux de réponses tronquées : la génération atteint le plafond sans token de fin ;
- exactitude : `extract_letter`, les réponses sans lettre extraite comptent fausses, et une réponse tronquée compte fausse même si une lettre en sort (raisonnement inachevé) ;
- longueur moyenne en tokens ;
- IC95 par bootstrap percentile (`eval.stats.bootstrap_ci`).

Fin de réponse : union de `generation_config.eos_token_id`, du `eos_token_id` du tokenizer et de `<|im_end|>` (le `config.json` de Qwen3.5 ne donne que `<|endoftext|>`) ; la liste retenue est écrite dans `summary.json`.

Commande (extra `ablation`, qui tire `transformers>=5` pour Qwen3.5 ; il est en conflit avec l'extra `eval`, borné `<5`, et ne s'installe pas avec lui ; pour un GPU, installer à part une roue torch CUDA, puis lancer avec `uv run --no-sync`, sinon uv remet la roue CPU du lock ; le run ci-dessous a tourné sur torch 2.6.0+cu124, pas sur la version du lock) :

```powershell
uv sync --extra ablation
uv run --extra ablation python scripts/run_antiloop.py --run-id <id> --n 100 --max-new-tokens 512 `
  --origin <slm-weight-editing>/models/hf/qwen3.5-base `
  --surgery <slm-weight-editing>/models/hf/qwen3.5-wsp `
  --directions-dir <dossier des layer15.npy / layer19.npy> --device cuda --dtype bfloat16
```

`--origin` et `--surgery` sont obligatoires (ou `ANTILOOP_ORIGIN` / `ANTILOOP_SURGERY`). La configuration (chemins, `n`, source des items, plafond, `thinking`, device, dtype, versions de torch et transformers) est figée dans `config.json` au premier lancement : une reprise sous une autre configuration est refusée.

Sorties : `data/runs/antiloop/<id>/<bras>.jsonl` (une ligne par item, reprise en relançant la même commande) et `summary.json` (configuration, dtype effectif et tokens de fin par modèle, mesures par bras ; un lancement partiel `--arms` garde les bras déjà mesurés).

## Comparaison avec slm-quality-bench

`slm-quality-bench` mesure aussi le bras chirurgie (GGUF édité servi par Ollama) et en tire un taux tronqué par piste et par catégorie (`summary.json`). Ses conditions diffèrent (Ollama, quantification Q8_0, jeux et plafond propres), donc ses chiffres ne se mélangent pas à ceux de ce script : ils servent de contrôle croisé. Pour les comparer :

1. lire le taux tronqué de la piste `think_on` de `qwen3.5:0.8b` et de `qwen3.5-wsp:0.8b` dans leurs `summary.json` (`slm-weight-editing report-gguf` / `compare` en donne l'écart avec IC95) ;
2. vérifier que le sens et l'ordre de grandeur de l'écart origine / chirurgie concordent avec ceux mesurés ici (transformers, bf16) ;
3. ne pas reporter dans un même tableau des taux mesurés sous un plafond de tokens différent.

## Résultats

Run `antiloop-full` : `n=100` items Global-MMLU-Lite (fr), `max_new_tokens=512`, `enable_thinking=True`, génération gloutonne, bf16, torch 2.6.0+cu124 et transformers 5.22.1, GPU RTX 3050 8 Go, de 1,7 à 3,2 items par minute selon le bras, GPU partagé avec d'autres mesures (3 h pour les quatre bras). Tokens de fin retenus : 248044 et 248046 pour les deux modèles. Aucun écart au protocole du squelette. Bornes : IC95 par bootstrap percentile.

| Bras | n | Tronquées (IC95) | Exactitude (IC95) | Non extraites | Tokens moyens |
|---|---|---|---|---|---|
| origin | 100 | 0,97 [0,93 ; 1,00] | 0,02 [0,00 ; 0,05] | 36 | 507,7 |
| surgery | 100 | 0,97 [0,93 ; 1,00] | 0,03 [0,00 ; 0,07] | 39 | 508,6 |
| ablation@0.5 | 100 | 0,96 [0,92 ; 0,99] | 0,03 [0,00 ; 0,07] | 42 | 507,1 |
| ablation@1 | 100 | 0,97 [0,93 ; 1,00] | 0,02 [0,00 ; 0,05] | 50 | 508,3 |

Données brutes : `data/runs/antiloop/antiloop-full/` (ignoré par git, une ligne par item par bras, plus `summary.json`).

## Lecture et limites

Constat principal : avec un plafond de 512 tokens et le raisonnement ouvert, 96 à 97 % des générations atteignent le plafond dans chaque bras, avec une longueur moyenne d'environ 508 tokens. Les quatre bras ne se distinguent pas : les IC95 se recouvrent largement (avec n=100, seul un écart d'environ 5 points ou plus serait visible), et l'exactitude (2 à 3 %) est proche de zéro parce qu'une réponse tronquée compte fausse. Cette mesure ne permet donc de conclure ni que la chirurgie réduit les boucles, ni que l'ablation à l'inférence (doses 0,5 et 1,0) a un effet comparable : à ce plafond, le critère sature avant de pouvoir distinguer une boucle d'un raisonnement simplement long. La mesure ne contredit pas non plus l'effet rapporté par le MIT, elle n'a pas la résolution pour le voir.

Ce que le résultat montre, malgré tout : à ce plafond, ni l'édition des poids ni l'ablation d'une direction n'ont d'effet détectable (au-delà d'environ 5 points) sur le taux de réponses tronquées de Qwen3.5-0.8B sur ces items. Dans tous les bras, presque aucun raisonnement ne se termine dans le budget ; la mesure ne dit pas si c'est une boucle ou un raisonnement simplement long.

Chirurgie (permanente) contre ablation (dosable, réversible) : l'avantage structurel de l'ablation (dose réglable, aucun poids modifié) reste à démontrer, faute d'effet mesurable ici. Le démonstrateur montre l'expression d'un comportement, jamais un ressenti.

Limites :

- plafond unique de 512 tokens, trop bas pour ce jeu en mode raisonnement ; c'est la cause probable du plafonnement des mesures, à vérifier par un run à plafond plus haut ;
- 100 items, un seul tirage, génération gloutonne ;
- modèle de 0,8 milliard de paramètres, direction issue de la chirurgie elle-même ;
- bf16 sur transformers avec le repli PyTorch des noyaux de Qwen3.5 (pas de `causal_conv1d` ni `flash-linear-attention`), plus lent ; son effet sur les résultats n'est pas mesuré ;
- le contrôle croisé avec `slm-quality-bench` (piste `think_on`, Ollama, Q8_0) n'a pas été fait ici.

Suite proposée (hors périmètre de cette mesure) : relancer avec un plafond plus haut (par exemple 2048 ou 4096 tokens) sur un n réduit, `--arms` permettant de cibler les bras, pour que la fraction de réponses terminées permette de comparer les bras ; ou utiliser un jeu d'items qui provoque des boucles.
