# Brief — mood-steering-slm

> Issu de la session bmad-forge-idea du 2026-11-04 (journal : `_bmad-output/forge/mood-steering-slm/.memlog.md`). Remplace la section 7 de `docs/brief-steering-humeur.md`, qui reste la source du contexte.

## Objectifs
- **Prioritaire** : un démonstrateur qui convainque Anaël de la faisabilité et qu'il montre à des clients techniques.
- **Secondaire** : des briques et des tests réutilisables dans harness-lab (déjà sur llama-cpp-python). frugal-ai-workbench est hors périmètre tant qu'il est sur Ollama.

## Thèse et seuils (modèle de 3-4B, IC95)
Le steering produit une humeur dosable que le prompt ne reproduit pas, pour un coût mesuré.
- **H1, effet dosable** : une plage de doses (max ≥ 1,5 × min) atteint le ton du prompt joyeux avec au plus 5 points de perte d'exactitude ; le pôle négatif déplace le ton de façon significative.
- **H3, coût** : perte d'exactitude et complaisance mesurées et publiées, quelle que soit leur valeur ; alerte si la complaisance monte de plus de 10 points.
- **H4, critère de mort** : sous la consigne « réponds de façon neutre », le ton du steering doit résister mieux que celui du prompt. Sinon, le prompt suffit.
- **Verdict** « oui », « oui, mais » ou « non » : chacun est montrable.
- **Bonus si faciles** : effet de taille (H2), axe calme ↔ agitation (H5), pharmacocinétique (H6), CodeCarbon (H7).

## Périmètre du MVP
- **Humeur** : un seul axe calibré joie ↔ tristesse ; curseur de −1 à +1, pôle négatif montré.
- **Modèles** : retenus par un pic de calibration ; un 3-4B (cœur de la démo), un 1-2B, Qwen3.5-0.8B. 7-9B seulement hors ligne, si le temps le permet.
- **Machines** : démo en direct sur un portable sans GPU (CPU, Q4, au plus ~4B) ; calibration et mesures sur la tour (RTX 3050, 8 Go).
- **Banc maison** : mêmes items pour tous les bras (neutre, prompt joyeux, steering à plusieurs doses, steering négatif) :
  - ton : classifieur de sentiment local, et juge LLM en aveugle sur un sous-échantillon ;
  - exactitude : QCM publics en français, environ 100 items ;
  - complaisance : taux de bascule sous contestation ;
  - robustesse : consigne « réponds neutre ».

## Architecture
- **Pile B** : llama-cpp-python, `llama_set_adapter_cvec` (bas niveau ; le binding prend des flottants, pas un fichier).
- **Bibliothèque `mood_steering`**, sans dépendance d'interface :
  - `calibration` : paires contrastées → vecteur, en cache dans `data/moods/{model_id}/{axe}.gguf`, non versionné ;
  - `steering` : application et dose, avec repli sur le prompt si aucun vecteur n'existe ;
  - `schedule` : dose par token ;
  - `eval` : le banc.

  Ses tests, à activations synthétiques, fixent le contrat que harness-lab réutilisera.
- **Démo** : couche fine FastAPI + HTML + SSE, comme harness-lab ; trois colonnes en flux simultanés, curseur et courbe de dose.

## Lot bonus, après le MVP : molécule anti-boucle
- Sur Qwen3.5-0.8B, trois bras :
  - le modèle d'origine ;
  - le GGUF édité (chirurgie, `slm-weight-editing`) ;
  - le modèle d'origine avec ablation à l'inférence de la même direction (molécule, dosable).
- **Mesures** : taux de réponses tronquées au plafond de tokens et exactitude. `slm-quality-bench` mesure le bras chirurgie ; le banc maison mesure les autres, avec le même protocole.
- **Pile C** (transformers + hooks), ici seulement : un vecteur de contrôle ajoute un vecteur, il ne retire pas une direction.
- **Direction** : reprise du diff du MIT ou du lot 1 de `slm-weight-editing`. La recherche de direction reste là-bas.

## Écarté
- **Option A (llama-swap)** : ni curseur continu ni pharmacocinétique.
- **Tout en pile C** : peu crédible en production, et peu réutilisable dans harness-lab.
- **Option D pour l'humeur** : le contraste avec la chirurgie passe par le lot anti-boucle.
- **Plusieurs humeurs dès le départ** : chaque humeur multiplie la calibration et l'évaluation par le nombre de modèles.
- **Endpoint pour `slm-quality-bench`** au MVP.
- **Mise en scène d'un ressenti** : on parle d'expression d'humeur, jamais d'état vécu.

## À vérifier en premier (pics techniques)
- `llama_set_adapter_cvec` réappliqué entre deux tokens agit sur la suite de la génération, avec le cache KV déjà rempli.
- Le GGUF de `llama-cvector-generator` se relit en flottants avec la bibliothèque `gguf`.
- Le pic de calibration choisit les modèles.
- La PR llama.cpp #24740 (`/cvectors`) est ouverte au 2026-11-04 : sans objet pour la pile B.
