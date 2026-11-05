# Rapport anti-boucle : chirurgie des poids ou ablation dosable

Statut : deux mesures. Run `antiloop-full` du 2026-11-05 (story bonus 6.2, SPEC CAP-9, 512 tokens) : le plafond sature tous les bras, la mesure ne discrimine pas. Relance du 2026-11-06 (action A2 de la rétro de l'epic 6, run `antiloop-2048-n50`, 2048 tokens, n=50) : l'ablation à dose 1,0 réduit les réponses tronquées de 16 points (IC95 apparié [6 ; 28]), les autres écarts ne sont pas détectables ; le contrôle croisé `slm-quality-bench` montre un écart d'environ 30 points pour la chirurgie sous Ollama à 4096 tokens (voir Résultats de la relance et Lecture).

## Question

Sur Qwen3.5-0.8B, le MIT retire une direction (couches 15 et 19) des poids pour faire cesser les boucles de raisonnement ([slm-weight-editing](https://github.com/Aliquanto3/slm-weight-editing)). La même direction, ablatée à l'inférence avec une dose `alpha`, donne-t-elle un effet comparable sur le taux de réponses tronquées, sans modifier les poids ? Le rapport compare la chirurgie (permanente) à la molécule (dosable, réversible).

## Méthode

Trois bras, même protocole : mêmes items et même ordre, génération gloutonne, même plafond `max_new_tokens` (512 par défaut ; 2048 pour la relance), même gabarit de chat (`enable_thinking=True` : le `<think>` est ouvert, car les boucles sont des boucles de raisonnement ; le gabarit Qwen3.5 le referme sinon et la sortie se limite à la lettre). Pile : transformers.

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

Écarts appariés : `uv run --extra eval python scripts/run_antiloop.py --run-id <id> --report` relit les JSONL (sans générer ni charger de modèle) et écrit `paired.json` : pour chaque bras, écart `origin − bras` sur les tronquées et sur l'exactitude (bootstrap apparié de `eval.stats`, 5000 tirages, graine 0), nombre de paires discordantes et p du test de McNemar exact. Les bornes d'IC sont des percentiles de bootstrap : elles bougent d'environ 0,02 selon le nombre de tirages et la graine.

Sorties : `data/runs/antiloop/<id>/<bras>.jsonl` (une ligne par item, reprise en relançant la même commande) et `summary.json` (configuration, dtype effectif et tokens de fin par modèle, mesures par bras ; un lancement partiel `--arms` garde les bras déjà mesurés).

## Comparaison avec slm-quality-bench

`slm-quality-bench` mesure aussi le bras chirurgie (GGUF édité servi par Ollama) et en tire un taux tronqué par piste et par catégorie (`summary.json`). Ses conditions diffèrent (Ollama, quantification Q8_0, jeux et plafond propres), donc ses chiffres ne se mélangent pas à ceux de ce script : ils servent de contrôle croisé. Pour les comparer :

1. lire le taux tronqué de la piste `think_on` de `qwen3.5:0.8b` et de `qwen3.5-wsp:0.8b` dans leurs `summary.json` (`slm-weight-editing report-gguf` / `compare` en donne l'écart avec IC95) ;
2. vérifier que le sens et l'ordre de grandeur de l'écart origine / chirurgie concordent avec ceux mesurés ici (transformers, bf16) ;
3. ne pas reporter dans un même tableau des taux mesurés sous un plafond de tokens différent.

Contrôle fait le 2026-11-06 (campagne `wsp-20261104`, voir Résultats de la relance).

## Résultats de la relance (2048 tokens, n=50)

Relance décidée par la rétro de l'epic 6 (constats E6-1 et E6-2) : le plafond de 512 tokens saturait tout. Règle fixée avant la mesure : un pilote de l'origine seule (`antiloop-pilote-2048`, n=10, 2048 tokens, environ 54 s par item) donne 0,60 de tronquées [0,30 ; 0,90] ; moins de 80 %, donc on reste à 2048 (sinon 4096).

Run `antiloop-2048-n50` : quatre bras, les 50 mêmes items Global-MMLU-Lite (fr), `max_new_tokens=2048`, `enable_thinking=True`, génération gloutonne, bf16, torch 2.6.0+cu124, transformers 5.22.1, GPU RTX 3050 8 Go utilisé seul. IC95 des taux : bootstrap percentile. Écarts : `--report` (bootstrap apparié, 5000 tirages, graine 0) et McNemar exact sur les paires discordantes (origine seule / bras seul).

| Bras | Tronquées (IC95) | Exactitude (IC95) | Non extraites | Tokens moyens |
|---|---|---|---|---|
| origin | 0,78 [0,66 ; 0,88] | 0,16 [0,06 ; 0,26] | 25 | 1738,8 |
| surgery | 0,70 [0,58 ; 0,82] | 0,26 [0,14 ; 0,38] | 24 | 1711,0 |
| ablation@0.5 | 0,70 [0,58 ; 0,82] | 0,24 [0,12 ; 0,36] | 28 | 1702,5 |
| ablation@1 | 0,62 [0,48 ; 0,76] | 0,32 [0,20 ; 0,44] | 28 | 1647,1 |

Écarts appariés `origin − bras` sur les tronquées (positif : le bras tronque moins) :

| Bras | Écart | IC95 apparié | Discordants (origine seule / bras seul) | McNemar exact p |
|---|---|---|---|---|
| ablation@0.5 | +0,08 | [−0,06 ; 0,22] | 8 / 4 | 0,39 |
| ablation@1 | +0,16 | [0,06 ; 0,28] | 9 / 1 | 0,021 |
| surgery | +0,08 | [−0,06 ; 0,22] | 8 / 4 | 0,39 |

Même lecture sur l'exactitude (`origin − bras`, négatif : le bras est meilleur) : ablation@0.5 −0,08 [−0,20 ; 0,04] (3 / 7, p 0,34) ; ablation@1 −0,16 [−0,28 ; −0,06] (1 / 9, p 0,021) ; surgery −0,10 [−0,22 ; 0,02] (3 / 8, p 0,23). L'exactitude suit les tronquées parce qu'une réponse tronquée compte fausse.

Contrôle croisé `slm-quality-bench` (campagne `wsp-20261104`, jeu `set-v3`, 8 usages, Ollama 0.40.14, plafond 4096 tokens et 2048 pour l'usage rag : plafonds différents de ceux du tableau ci-dessus, donc tableau séparé), piste `think_on`, global, vérifié dans `summary.json` :

| Modèle | Tronquées / évaluées | Taux |
|---|---|---|
| `qwen3.5:0.8b` | 558 / 765 | 0,729 |
| `qwen3.5-wsp:0.8b` | 333 / 770 | 0,432 |

Écart origine − chirurgie : +0,297, IC95 [0,250 ; 0,344] (deux proportions indépendantes, approximation normale ; Newcombe [0,249 ; 0,343]). Les `scores/*.jsonl` n'ont pas d'indicateur de troncature par unité : l'appariement des questions est impossible pour ce critère, l'IC est celui de deux proportions et l'ignore. Piste `think_off` : 49 / 772 (0,063) contre 43 / 773 (0,056), écart +0,008 [−0,016 ; 0,031], sans écart. Sur la piste `think_on`, l'écart varie beaucoup selon l'usage (de 0 point à plus de 50 points ; un usage ne tronque jamais).

## Résultats du run historique `antiloop-full` (512 tokens, n=100)

Run `antiloop-full` : `n=100` items Global-MMLU-Lite (fr), `max_new_tokens=512`, `enable_thinking=True`, génération gloutonne, bf16, torch 2.6.0+cu124 et transformers 5.22.1, GPU RTX 3050 8 Go, de 1,7 à 3,2 items par minute selon le bras, GPU partagé avec d'autres mesures (3 h pour les quatre bras). Tokens de fin retenus : 248044 et 248046 pour les deux modèles. Aucun écart au protocole du squelette. Bornes : IC95 par bootstrap percentile.

| Bras | n | Tronquées (IC95) | Exactitude (IC95) | Non extraites | Tokens moyens |
|---|---|---|---|---|---|
| origin | 100 | 0,97 [0,93 ; 1,00] | 0,02 [0,00 ; 0,05] | 36 | 507,7 |
| surgery | 100 | 0,97 [0,93 ; 1,00] | 0,03 [0,00 ; 0,07] | 39 | 508,6 |
| ablation@0.5 | 100 | 0,96 [0,92 ; 0,99] | 0,03 [0,00 ; 0,07] | 42 | 507,1 |
| ablation@1 | 100 | 0,97 [0,93 ; 1,00] | 0,02 [0,00 ; 0,05] | 50 | 508,3 |

Données brutes : `data/runs/antiloop/antiloop-full/` (ignoré par git, une ligne par item par bras, plus `summary.json`).

## Lecture et limites

Lecture de la relance (2048 tokens, n=50) :

- À 2048 tokens, 78 % des générations de l'origine atteignent encore le plafond : la saturation a baissé (97 % à 512) mais reste forte, et la mesure ne sépare toujours pas une boucle d'un raisonnement long. Avec n=50, on ne voit que des écarts d'environ 15 points.
- Seule l'ablation à dose 1,0 a un IC95 apparié qui exclut 0, sur les tronquées (+16 points [6 ; 28]) comme sur l'exactitude (+16 points). Le McNemar exact va dans le même sens (9 contre 1, p = 0,021) ; non corrigé des trois comparaisons, il ne passe pas le seuil de Bonferroni (0,017). À lire comme un indice, pas comme une preuve.
- L'ablation à dose 0,5 et la chirurgie vont dans le même sens (+8 points) sans écart détectable ici. L'ordre des doses (0,5 puis 1,0) est cohérent avec un effet dose, mais sur deux doses et un seul tirage.
- La chirurgie va dans le même sens que `slm-quality-bench`, mais sans écart détectable ici. Le contrôle croisé montre un écart net (environ 30 points, IC95 [25 ; 34]) à 4096 tokens sous Ollama (GGUF Q8_0, autre jeu, autres usages). Les conditions diffèrent trop (plafond, pile, quantification, jeu) pour dire que la relance reproduit ou infirme cet écart : +8 points [−6 ; 22] est compatible avec lui sans le confirmer.
- Ce que la relance permet de conclure : l'ablation de la direction à dose 1,0 diminue vraisemblablement les réponses tronquées de Qwen3.5-0.8B sur ces items. Ce qu'elle ne permet pas : comparer ablation et chirurgie (leurs IC se recouvrent), établir un effet de l'ablation à 0,5, dire que les réponses évitées étaient des boucles, ni généraliser hors de ces 50 items.

Lecture du run historique `antiloop-full` (512 tokens, n=100) : 96 à 97 % de réponses tronquées dans chaque bras, longueur moyenne d'environ 508 tokens, quatre bras indiscernables (avec n=100, seul un écart d'environ 5 points ou plus serait visible) et exactitude (2 à 3 %) proche de zéro. À ce plafond, le critère sature avant de distinguer une boucle d'un raisonnement long ; la mesure ne contredisait pas l'effet rapporté par le MIT, elle n'avait pas la résolution pour le voir. C'est ce constat qui a motivé la relance.

Chirurgie (permanente) contre ablation (dosable, réversible) : l'avantage structurel de l'ablation (dose réglable, aucun poids modifié) n'est pas démontré par ces mesures : un indice à dose 1,0, rien de détectable à 0,5. Le démonstrateur montre l'expression d'un comportement, jamais un ressenti.

Limites :

- plafond de 2048 tokens encore trop bas pour 78 % des générations de l'origine ; 4096 (celui de `slm-quality-bench`) serait l'étape suivante, mais plus long (le pilote donnait environ 54 s par item à 2048) ;
- 50 items, un seul tirage, génération gloutonne ; trois comparaisons sans correction de multiplicité ;
- modèle de 0,8 milliard de paramètres, direction issue de la chirurgie elle-même ;
- bf16 sur transformers avec le repli PyTorch des noyaux de Qwen3.5 (pas de `causal_conv1d` ni `flash-linear-attention`), plus lent ; son effet sur les résultats n'est pas mesuré ;
- contrôle croisé fait, mais non comparable chiffre à chiffre : Ollama, Q8_0, autre jeu, autre plafond, et pas d'indicateur de troncature par unité pour apparier.

Suite proposée (hors périmètre) : plafond de 4096 et n plus grand pour l'ablation à dose 1,0 contre l'origine ; ou un jeu d'items qui provoque des boucles.
