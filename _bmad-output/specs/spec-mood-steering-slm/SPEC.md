---
id: SPEC-mood-steering-slm
companions:
  - hypotheses-et-seuils.md
  - stack.md
sources:
  - ../../planning-artifacts/brief.md
  - ../../../docs/brief-steering-humeur.md
---

> **Contrat canonique.** Ce SPEC et ses `companions:` forment le contrat complet de ce qu'il faut construire, tester et valider. Les sources ne servent qu'à la traçabilité.

# mood-steering-slm — humeur de SLM par steering d'activations

## Why

Vision et opportunité. Les « drogues pour ChatGPT » du marché ne sont que des prompts de jeu de rôle. Anaël veut savoir, puis montrer à des clients techniques, si un vecteur ajouté aux activations à l'inférence (la « molécule ») produit une humeur dosable que le prompt ne reproduit pas, et à quel coût en exactitude et en complaisance. Objectif secondaire : des briques et des tests réutilisables dans harness-lab (déjà sur llama-cpp-python).

## Capabilities

- **CAP-1**
  - **intent :** Le système calibre, pour un modèle donné, un vecteur joie ↔ tristesse à partir de paires de prompts contrastées, le met en cache et le relit en flottants.
  - **success :** Un vecteur calibré sur un GGUF est rangé dans `data/moods/{model_id}/{axe}.gguf` ; un second appel le relit sans recalibrer ; la relecture rend une couche → tableau de flottants de la dimension cachée du modèle.
- **CAP-2**
  - **intent :** L'utilisateur règle une dose continue de −1 à +1 appliquée au modèle chargé sans le recharger ; sans vecteur pour ce modèle, le système se replie sur le prompt.
  - **success :** Deux générations successives à des doses différentes ne rechargent pas le modèle ; dose 0 = sortie du bras neutre ; modèle sans vecteur → bras prompt, signalé.
- **CAP-3**
  - **intent :** La dose peut varier token par token pendant une génération.
  - **success :** Un profil de dose (constant, ou montée puis décroissance) est réappliqué entre deux tokens et change la suite du texte, cache KV déjà rempli (validé par le pic 1).
- **CAP-4**
  - **intent :** Un banc maison passe les mêmes items à tous les bras (neutre, prompt joyeux, steering à plusieurs doses, steering négatif) et mesure ton, exactitude, complaisance et robustesse.
  - **success :** Une commande produit, par bras et par modèle, les quatre mesures avec IC95, sur ~100 QCM publics en français ; protocole détaillé dans `hypotheses-et-seuils.md`.
- **CAP-5**
  - **intent :** Le projet tranche la thèse par un verdict publié.
  - **success :** Un rapport dans `docs/` donne, sur le modèle 3-4B, H1, H3 et H4 contre leurs seuils, et un verdict « oui », « oui, mais » ou « non ».
- **CAP-6**
  - **intent :** Un client technique voit, en direct, le même prompt rendu en neutre, en prompt joyeux et en steering, avec curseur de dose.
  - **success :** Sur le portable sans GPU, trois colonnes se remplissent en flux simultanés ; déplacer le curseur change la dose de la génération suivante ; la courbe de dose s'affiche ; le pôle négatif est montrable.
- **CAP-7**
  - **intent :** Un pic de calibration choisit les modèles du projet.
  - **success :** Une décision consignée retient un 3-4B (cœur de la démo), un 1-2B et Qwen3.5-0.8B, avec pour chacun la couche visée et la plage de doses utile observée.
- **CAP-8** *(bonus, si facile)*
  - **intent :** Le banc mesure en plus l'effet de taille (H2), l'axe calme ↔ agitation (H5), la pharmacocinétique (H6) et le coût énergétique (H7, CodeCarbon).
  - **success :** Chaque hypothèse bonus traitée a sa section dans le rapport ; une hypothèse non traitée y est marquée « non traitée ».
- **CAP-9** *(bonus, après le MVP)*
  - **intent :** Sur Qwen3.5-0.8B, le projet compare la « chirurgie » (GGUF édité) et la « molécule » (ablation dosable à l'inférence) contre les boucles de génération.
  - **success :** Trois bras (origine, chirurgie, ablation) mesurés avec le même protocole : taux de réponses tronquées au plafond de tokens et exactitude, publiés dans un rapport.

## Constraints

- Pile B pour l'humeur : llama-cpp-python, `llama_set_adapter_cvec` (bas niveau, prend des flottants et non un fichier). Détail dans `stack.md`.
- Un vecteur est propre au GGUF et à la quantification servis ; il est calibré sur ce même fichier. Ollama est exclu (pas d'accès aux activations).
- La bibliothèque `mood_steering` n'a aucune dépendance d'interface ; ses tests, à activations et vecteurs synthétiques, n'utilisent ni modèle réel ni téléchargement : ils fixent le contrat que harness-lab réutilisera.
- Dépôt public : ni poids, ni GGUF (vecteurs compris), ni jeux de données, ni sorties volumineuses dans un commit (`data/`, `models/` ignorés).
- Démo en direct sur portable sans GPU (CPU, Q4, au plus ~4B) ; calibration et mesures sur la tour (RTX 3050, 8 Go).
- Pile C (transformers + hooks) seulement dans le lot anti-boucle : un vecteur de contrôle ajoute une direction, il n'en retire pas.
- Les classifieurs de ton et le juge vivent dans un extra optionnel `eval` (transformers, torch, anthropic) que le cœur de `mood_steering` n'importe pas. Choix des mesures dans `hypotheses-et-seuils.md`.
- Registre : le démonstrateur montre l'expression d'une humeur, jamais un ressenti, dans les docs comme dans l'interface.
- Un seul axe calibré au MVP : joie ↔ tristesse.

## Non-goals

- Option A (llama-swap) : ni curseur continu ni pharmacocinétique.
- Tout en pile C : peu crédible en production, peu réutilisable dans harness-lab.
- Option D (chirurgie) pour l'humeur : le contraste passe par le lot anti-boucle.
- Plusieurs humeurs au MVP (calme ↔ agitation reste un bonus).
- Endpoint pour `slm-quality-bench` au MVP.
- Recherche de la direction anti-boucle : reprise du diff du MIT ou du lot 1 de `slm-weight-editing`, pas refaite ici.
- Intégration dans frugal-ai-workbench tant qu'il est sur Ollama ; produit pour harness-lab.
- Modèles 7-9B en démo (hors ligne seulement, si le temps le permet).

## Success signal

- Le rapport de verdict sur le 3-4B est publié avec IC95, et la démo trois colonnes tourne en direct sur le portable CPU devant un client technique, pôle négatif compris. Un « non » publié compte comme un succès.

## Assumptions

- Si le pic 1 échoue, la dose est fixée par requête (le curseur entre deux requêtes reste) ; CAP-3 et H6 sont abandonnés.
- Les ~100 paires contrastées, rédigées à la main, sont versionnables : texte d'auteur, pas un jeu de données.
- « Atteindre le ton du prompt » : l'IC95 du score de ton moyen du bras steering recouvre celui du bras prompt joyeux.

- Le run autonome tourne sur la tour : les pics et la campagne s'exécutent sans Anaël ; seul le test sur le portable reste humain et ne bloque pas.

## Open Questions

- Taille du sous-échantillon soumis au juge : à fixer en story 3.2 d'après le coût estimé.
