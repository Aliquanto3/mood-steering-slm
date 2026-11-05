# mood-steering-slm v0.1.0 (brouillon)

Premier lot : un démonstrateur d'expression d'humeur d'un petit modèle de langage par steering d'activations à l'inférence, comparé à un prompt de jeu de rôle. Il montre l'expression d'une humeur dans le texte, jamais un ressenti.

## Livré

- **Epic 1, steering.** Bibliothèque `mood_steering` : paires contrastées joie/tristesse, génération du vecteur de contrôle (llama.cpp, méthode `mean`), lecture et validation du GGUF, `Steerer` dosable de −1 à +1 sur une plage de couches, sur llama-cpp-python (CPU ou GPU).
- **Epic 2, modèles et pics.** Choix des modèles par balayage (Qwen3.5 4B, 2B et 0.8B ; Llama-3.2 en remplaçant), plages de couches et `max_scale` par modèle (`docs/pics/`).
- **Epic 3, évaluation.** Banc de mesure : ton (classifieur, polarité, juge en aveugle), exactitude sur QCM, complaisance, ton sous consigne neutre, sorties dégénérées ; rapports `docs/rapport-verdict.md` et `docs/rapport-modeles-secondaires.md`, test anti-boucle.
- **Epic 4, démo.** Serveur FastAPI avec flux SSE à trois colonnes (neutre, prompt joyeux, steering), page avec curseur de dose et courbe de dose, repli sans vecteur ; guide `docs/demo.md`.

## Verdict (Qwen3.5-4B-Q4_K_M)

- H1 non tenue : seule la dose +1 égale le ton du prompt joyeux.
- H4 oui, nettement : sous consigne neutre, le prompt joyeux tombe de 0,75 à 0,05 en ton ; le steering +1 reste à 0,58.
- H3 : pas d'alerte sur l'exactitude pour les doses positives ; puissance de la complaisance tout juste sous la cible.
- Détail et intervalles : [`rapport-verdict.md`](rapport-verdict.md).

## Limites

- Pôle négatif : 1 % de sorties dégénérées à −0,43, 19 % à −0,71, 49 % à −1 (boucles, refus d'assistant). Seule la dose −0,43 est propre.
- Un seul axe (joie/tristesse), un modèle de cœur, génération gloutonne.
- Campagne mesurée sur GPU (roue CUDA) ; la démo tourne sur CPU : répétition du 2026-11-05, environ 5 tokens par seconde et par colonne (trois colonnes en parallèle), 21 à 26 s pour 128 tokens, pôle négatif compris.
- Poids, vecteurs et données ne sont pas dans le dépôt (voir les prérequis de `docs/demo.md`).

## Étape humaine restante

Test de `docs/demo.md` sur le portable par Anaël, pour confirmer la vitesse réelle. Cette étape ne bloque pas la release.
