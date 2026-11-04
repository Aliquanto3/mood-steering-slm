# Pile technique

## Pile B (humeur)

- llama-cpp-python, API C bas niveau `llama_set_adapter_cvec` : le binding prend un tableau de flottants (couches × dimension cachée), pas un fichier. La PR llama.cpp #24740 (`/cvectors`, ouverte au 2026-11-04) est sans objet ici.
- Génération des vecteurs : `llama-cvector-generator` (ACP ou `--method mean`, CPU ou GPU) sur le GGUF servi ; sortie GGUF relue en flottants avec la bibliothèque Python `gguf`.
- Point de départ conseillé par llama.cpp : couches > 10, échelle ~0,8. Calibration sans gradient : ~100 paires contrastées, quelques minutes, une fois par couple modèle × axe.
- Changer de vecteur à chaud dans llama-server n'est pas natif (issue #10685) : d'où l'in-process.

## Bibliothèque `mood_steering`

| Module | Rôle |
|---|---|
| `calibration` | Paires contrastées → vecteur ; cache `data/moods/{model_id}/{axe}.gguf`, non versionné. |
| `steering` | Application du vecteur et dose ; repli sur le prompt si aucun vecteur. |
| `schedule` | Dose par token. |
| `eval` | Le banc (bras, mesures, IC95, rapport). |

## Démo

Couche fine FastAPI + HTML + SSE, comme harness-lab : trois colonnes en flux simultanés (neutre, prompt joyeux, steering), curseur de dose, courbe de dose.

## Pile C (lot anti-boucle seulement)

transformers + hooks PyTorch pour l'ablation d'une direction à l'inférence. Direction reprise du diff du MIT ou du lot 1 de `slm-weight-editing`.

## Outillage

uv (`uv run python`), ruff, pytest ; CI sur Linux, macOS et Windows.
