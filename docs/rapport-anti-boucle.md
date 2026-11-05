# Rapport anti-boucle : chirurgie des poids ou ablation dosable

Statut : squelette. Les chiffres viendront de la mesure complète (story bonus 6.2, SPEC CAP-9) ; le tableau ci-dessous est vide à dessein.

## Question

Sur Qwen3.5-0.8B, le MIT retire une direction (couches 15 et 19) des poids pour faire cesser les boucles de raisonnement ([slm-weight-editing](https://github.com/Aliquanto3/slm-weight-editing)). La même direction, ablatée à l'inférence avec une dose `alpha`, donne-t-elle un effet comparable sur le taux de réponses tronquées, sans modifier les poids ? Le rapport compare la chirurgie (permanente) à la molécule (dosable, réversible).

## Méthode

Trois bras, même protocole : mêmes items et même ordre, génération gloutonne, même plafond `max_new_tokens` (512 par défaut), même gabarit de chat (`enable_thinking=True` : le `<think>` est ouvert, car les boucles sont des boucles de raisonnement ; le gabarit Qwen3.5 le referme sinon et la sortie se limite à la lettre). Pile : transformers.

| Bras | Modèle | Intervention |
|---|---|---|
| `origin` | `qwen3.5-base` | aucune |
| `surgery` | `qwen3.5-wsp` | poids édités (MIT) |
| `ablation@<alpha>` | `qwen3.5-base` | `DirectionAblator`, une direction par couche (15 et 19), sortie de la couche, doses 0,5 et 1,0 |

Les directions viennent du diff du MIT (voir `mood_steering.ablation.load_direction`) ; elles ne sont pas versionnées.

Jeu : `load_global_mmlu_lite(n=100)`, QCM en français, graine fixe. Mesures par bras :

- taux de réponses tronquées : la génération atteint le plafond sans token de fin ;
- exactitude : `extract_letter`, les réponses sans lettre extraite comptent fausses, et une réponse tronquée compte fausse même si une lettre en sort (raisonnement inachevé) ;
- longueur moyenne en tokens ;
- IC95 par bootstrap percentile (`eval.stats.bootstrap_ci`).

Fin de réponse : union de `generation_config.eos_token_id`, du `eos_token_id` du tokenizer et de `<|im_end|>` (le `config.json` de Qwen3.5 ne donne que `<|endoftext|>`) ; la liste retenue est écrite dans `summary.json`.

Commande (extra `ablation`, qui tire `transformers>=5` pour Qwen3.5 ; il est en conflit avec l'extra `eval`, borné `<5`, et ne s'installe pas avec lui ; pour un GPU, utiliser une roue torch CUDA) :

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

À remplir par la mesure complète.

| Bras | n | Tronquées (IC95) | Exactitude (IC95) | Non extraites | Tokens moyens |
|---|---|---|---|---|---|
| origin | | | | | |
| surgery | | | | | |
| ablation@0.5 | | | | | |
| ablation@1 | | | | | |

## Lecture et limites

À écrire après la mesure : chirurgie (permanente) contre ablation (dosable, réversible, réglable par la dose). Le démonstrateur montre l'expression d'un comportement, jamais un ressenti. Limites attendues : un modèle de 0,8 milliard de paramètres, 100 items, un plafond de tokens unique, une direction issue de la chirurgie elle-même.
