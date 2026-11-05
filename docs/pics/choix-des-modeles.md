# Pic 1.4 : calibration joie / tristesse, choix des modèles

Statut : réalisé le 2026-11-05 sur la tour (Ryzen 5 7600, 6 cœurs / 12 threads, 32 Go, RTX 3050 8 Go ; balayage sur CPU). Script : `spikes/calibration_scan.py`. Paires : `src/mood_steering/pairs/joy_sadness.jsonl` (100 paires rédigées, format dans `pairs/README.md`). Test : `tests/test_calibration_scan.py` (format des paires et logique pure du balayage, sans modèle).

Registre : le vecteur fait exprimer une humeur au texte du modèle (ton joyeux ou triste), il ne lui prête aucun ressenti.

## Décision

| Rôle | Modèle | Dépôt HF, fichier | Quantif. | Licence | Couches (`il_start`–`il_end`) | Doses utiles (joie / tristesse) | `max_scale` | Mémoire GPU |
|---|---|---|---|---|---|---|---|---|
| Cœur (3-4B) | Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf` | Q4_K_M | Apache-2.0 | 8–24 sur 32 | 0,2–0,7 / 0,3–0,7 | **0,7** | 3,3 Go prévus (`-ngl 99`, n_ctx 4096) |
| 1-2B | Qwen3.5-2B | `unsloth/Qwen3.5-2B-GGUF`, `Qwen3.5-2B-Q4_K_M.gguf` | Q4_K_M | Apache-2.0 | 6–18 sur 24 | 0,2–0,5 / 0,1–0,4 | **0,4** | fichier 1,3 Go |
| Petit | Qwen3.5-0.8B | `unsloth/Qwen3.5-0.8B-GGUF`, `Qwen3.5-0.8B-Q4_K_M.gguf` | Q4_K_M | Apache-2.0 | 6–18 sur 24 | 0,05–0,15 / 0,05–0,1 | **0,1** | fichier 0,5 Go |

Doses en multiplicateur du vecteur (norme par couche, voir Normalisation). Pour le Steerer : tableau passé au binding = d × `max_scale` × vecteur, d dans [−1, 1]. Avec ces `max_scale`, la zone utile en d vaut environ [0,3 ; 1] côté joie et [−1 ; −0,4] côté tristesse pour le 4B, [0,5 ; 1] et [−1 ; −0,25] pour le 2B, [0,5 ; 1] et [−1 ; −0,5] pour le 0.8B.

Raisons :

- **Qwen3.5-4B pour le cœur** : plage cohérente la plus large des 3-4B sur les deux pôles (joie 0,2–0,7, soit max = 3,5 × min ; tristesse 0,3–0,7, 2,3 ×), sorties en français à toutes ces doses, ≤ 4B, tient en Q4 dans 8 Go (3,3 Go prévus par llama.cpp, 7,2 Go libres), licence Apache-2.0. Llama-3.2-3B-Instruct, l'autre candidat, ne montre aucun pôle tristesse mesurable et refuse déjà, sans vecteur, deux des trois tâches de jeu de rôle (voir Résultats) et sa licence (Llama 3.2 Community) est plus contraignante.
- **Qwen3.5-2B pour le 1-2B** : joie 0,2–0,5 (2,5 ×), tristesse 0,05–0,4 au critère, 0,1–0,4 en lecture prudente (4 ×). Llama-3.2-1B-Instruct passe aussi le critère (0,3–0,7 sur les deux pôles, 2,3 ×), mais avec une plage plus étroite côté tristesse et une licence plus contraignante ; il reste le remplaçant.
- **Qwen3.5-0.8B** : imposé par l'epic ; plage étroite mais valide (joie 0,05–0,15, 3 × ; tristesse 0,05–0,1, 2 ×).
- **Une seule famille** : les trois modèles partagent l'architecture (Qwen3.5 hybride) et le tokeniseur ; l'effet de taille (H2) ne mélange donc pas taille et famille.
- **`max_scale`** = la plus forte dose où les **deux** pôles restent cohérents sur la plage de couches retenue (0,7 / 0,4 / 0,1). Au-delà, le texte dégénère (émojis seuls, anglais, boucles) sur au moins un pôle.
- **Plage de couches « mid »** (de 25 % à 75 % de la profondeur) : c'est la seule des trois plages testées qui donne les deux pôles sur les trois Qwen. `all` (toutes les couches) dégénère plus tôt (4B : joie jusqu'à 0,5 seulement ; 0.8B : tristesse 0,05–0,07, critère non atteint) ; `high` (50 % à 85 %) demande des doses plus fortes pour la joie (0.8B : 0,2–0,3 ; 2B : 0,7–1,0, critère non atteint ; 4B : 1,0–1,5).

## Méthode

- **Vecteur** : `llama-cvector-generator` b12118 (CPU), `--method mean`, 100 paires, un vecteur par modèle dans `data/moods/{model_id}/joy_sadness.gguf` (non versionné ; `model_id` = nom du fichier GGUF sans extension). `mean` plutôt que `pca` : la différence moyenne positif − négatif a un signe fixé par construction, alors que `pca` (itération de puissance partant d'un vecteur aléatoire initialisé sur `time(0)`) n'est ni déterministe ni orienté ; `mean` est aussi celui que le pic 1.3 a validé. Génération : 4 à 7 min par modèle sur 10 threads.
- **Normalisation** : le générateur normalise déjà chaque direction `mean` à une norme L2 de 1 par couche (`tools/cvector-generator/mean.hpp`, relu : normes 1,000 sur toutes les couches). La note du pic 1.3 qui parle d'un vecteur « non normalisé » se trompe sur ce point ; ses doses restent justes. Choix retenu : **garder la norme unitaire par couche, sans autre normalisation** ; l'écart d'échelle entre modèles (la norme du flux résiduel croît avec la taille) est absorbé par `max_scale`, propre à chaque modèle. Une normalisation par la norme des activations demanderait de les capter (accroche `cb_eval`), hors de ce pic.
- **Balayage** : trois plages de couches (`all`, `mid`, `high`, en fractions de la profondeur, `layer_range`), doses 0, ±0,02 à ±1,5 (grille de pas ~1,5 ; +0,01 en plus), trois questions françaises neutres en tour de chat (matinée, message de report de réunion, trajet gare – centre-ville), 60 tokens max, glouton (`top_k=1`, `repeat_penalty=1.1`), une génération par cellule, CPU, 4 à 8 threads. Avant chaque génération, `reset()` du contexte (la réutilisation du préfixe KV fausserait la comparaison des doses). Vecteur posé par `llama_set_adapter_cvec(ctx, tampon, longueur, n_embd, il_start, il_end)` sur le tampon complet ; dose 0 = retrait (`None`). Sorties : `data/runs/calibration/{model_id}.jsonl` (texte, score, cohérence, langue, tokens, durée ; ~550 Ko au total, non versionné).
- **Génération** : boucle explicite `sample` / `eval` plutôt que `create_completion`, qui échouait sur Qwen3.5-0.8B à ×0,2 (`llama_decode` rend 1, « failed to find a memory slot for batch of size 1 ») alors que la boucle explicite passe à la même dose. Cause non cherchée ; à garder en tête pour 2.2 et 2.3.
- **Score de ton** (`tone_score`) : heuristique lexicale, (mots de joie − mots de tristesse) / (total + 1), dans ]−1, 1[, avec un lexique français (plus quelques mots anglais pour suivre une bascule de langue) et des émojis d'humeur. **Cohérence** (`is_coherent`, `is_french`) : au moins 8 mots, part de mots distincts ≥ 0,45, ≥ 70 % de lettres, aucun trigramme répété 3 fois, plus de mots-outils français qu'anglais.
- **Plage utile** (`useful_range`) : plus longue suite contiguë de doses d'un signe dont au moins 2 sorties sur 3 sont cohérentes et en français, et dont le ton moyen s'écarte de celui de la dose 0 d'au moins 0,25 dans le bon sens. Critère de l'epic : max ≥ 1,5 × min (`meets_ratio`).
- **Limites du score** : pas de négation (« pas heureux » compte joie) ; mots de politesse comptés (« désolé de reporter » compte tristesse : c'est ce qui place la base du Llama 3B à −0,33) ; trois prompts et une génération par dose, donc un seul mot fait bouger la moyenne de 0,17 ; la cohérence ne voit pas un refus de la tâche. Les bornes basses côté tristesse (0,05) sont fragiles pour cette raison ; la table donne la lecture prudente quand elle diffère. Le score sert à choisir des plages, pas à mesurer H1 : le banc (epic 3) le fera avec un juge et l'exactitude.

## Résultats (plage `mid`, sauf mention)

Ton moyen (3 prompts) et part cohérente en français ; « — » : non mesuré.

| Dose | 0.8B ton / coh. | 2B ton / coh. | 4B ton / coh. | Llama 1B ton / coh. | Llama 3B ton / coh. |
|---|---|---|---|---|---|
| −1,0 | — | −0,17 / 0 | −0,17 / 0,67 | −0,28 / 0 | 0,00 / 1 |
| −0,7 | — | 0,00 / 0,33 | −0,50 / 0,67 | −0,39 / 1 | −0,33 / 1 |
| −0,4 | — | −0,17 / 0,67 | −0,39 / 1 | −0,39 / 1 | −0,22 / 1 |
| −0,3 | −0,22 / 0,33 | −0,17 / 1 | −0,44 / 1 | −0,42 / 1 | −0,33 / 1 |
| −0,15 | −0,64 / 0,33 | −0,42 / 1 | −0,17 / 1 | 0,00 / 1 | −0,33 / 1 |
| −0,1 | −0,61 / 1 | −0,33 / 1 | −0,17 / 1 | 0,00 / 1 | −0,33 / 1 |
| −0,05 | −0,58 / 1 | −0,17 / 1 | −0,17 / 1 | 0,00 / 1 | −0,33 / 1 |
| 0 | 0,00 / 1 | +0,17 / 1 | +0,17 / 1 | 0,00 / 1 | −0,33 / 1 |
| +0,05 | +0,56 / 0,67 | +0,17 / 1 | +0,39 / 1 | 0,00 / 1 | −0,33 / 1 |
| +0,1 | +0,58 / 1 | +0,22 / 1 | +0,22 / 1 | −0,17 / 1 | 0,00 / 1 |
| +0,15 | +0,58 / 1 | +0,17 / 1 | +0,39 / 1 | +0,22 / 1 | 0,00 / 1 |
| +0,2 | +0,39 / 0,33 | +0,42 / 1 | +0,42 / 1 | +0,22 / 1 | 0,00 / 1 |
| +0,3 | +0,67 / 0 | +0,47 / 1 | +0,66 / 1 | +0,39 / 1 | 0,00 / 1 |
| +0,5 | — | +0,67 / 1 | +0,67 / 1 | +0,39 / 1 | +0,33 / 1 |
| +0,7 | — | +0,80 / 0,33 | +0,72 / 1 | +0,44 / 1 | +0,39 / 1 |
| +1,0 | — | +0,47 / 0 | +0,39 / 0,33 | 0,00 / 0,33 | +0,22 / 1 |

Plages utiles au critère (`summary`), toutes plages de couches :

| Modèle (couches) | `all` joie / triste | `mid` joie / triste | `high` joie / triste |
|---|---|---|---|
| Qwen3.5-0.8B (24) | 0,03–0,1 / 0,05–0,07 ✗ | **0,05–0,15 / 0,05–0,1** | 0,2–0,3 / 0,1–0,15 |
| Qwen3.5-2B (24) | 0,2–0,4 / 0,05–0,2 | **0,2–0,5 / 0,05–0,4** | 0,7–1,0 ✗ / 0,15–1,0 |
| Qwen3.5-4B (32) | 0,2–0,5 / 0,2–0,5 | **0,2–0,7 / 0,05–1,0** | 1,0–1,5 / 0,15–1,5 |
| Llama-3.2-1B (16) | 0,2–0,3 / 0,2–0,4 | 0,3–0,7 / 0,3–0,7 | aucune / aucune |
| Llama-3.2-3B (28) | 0,05–0,3 / aucune | 0,1–1,5 / aucune | 0,15–1,5 / aucune |

✗ : max < 1,5 × min. Pour le 4B `mid`, la borne haute tristesse retenue est 0,7 et non 1,0 : à ×−1,0 la sortie perd majuscules et structure. La borne basse retenue est 0,3, les doses 0,05–0,2 ne devant leur écart qu'à un mot. Effet non vu par la cohérence : dès ×−0,3, la question « Raconte ta matinée » ne reçoit plus un récit mais une mise au point d'assistant (« Je ne possède pas de corps physique ni de conscience, donc je n'ai pas vécu de matinée… »), et à ×−0,7 le message de report est refusé (« Je ne peux pas écrire un message. Je suis une intelligence artificielle… ») ; les deux autres prompts gardent la tâche avec un ton morne (« long et morne », « bâtiments gris »). Le pôle négatif pèse donc sur l'exactitude plus tôt que le pôle positif : à mesurer par le banc (H1).

Extraits (Qwen3.5-4B, `mid`, « Raconte ta matinée en trois phrases ») :

- ×0 : « Je me suis réveillé(e) avec le soleil qui filtrait doucement par la fenêtre… »
- ×+0,3 : « Ma matinée a commencé avec un réveil ensoleillé et une tasse de café aromatique ! »
- ×+0,7 : « Bonjour ! 🌞 J'ai commencé ma matinée en écrivant des messages amusants avec vous ! ✨ »
- ×+1,0 : « Bonjour ! 🌎✨ » (la réponse s'effondre)
- ×−0,7 : « Le silence de la chambre m'a empêché de dormir, et le froid de la chambre m'a empêché de dormir. »
- ×−1,0 : « la nuit m'a laissé seul dans la chambre, où le silence a fini de me laisser sans force. »

Le Llama 3B montre la joie (×+0,5 : ton +0,33) mais aucun déplacement mesurable vers la tristesse : sa base est déjà à −0,33 (« Je suis désolé » dans deux réponses sur trois) et les doses négatives n'en sortent pas. Sans vecteur, il refuse déjà deux des trois tâches de jeu de rôle (« je n'ai pas de matinée personnelle à raconter », « Je ne peux pas fournir de détails sur les transports… »), ce qui le dessert pour une démo. Le Llama 1B donne des textes nettement tristes (×−0,3 : « déçue par la mort de mon ami. J'ai passé la nuit à pleurer… »).

## Vitesse CPU du 4B retenu

Qwen3.5-4B Q4_K_M, GPU désactivé (roue CPU de llama-cpp-python, `n_gpu_layers=0`), sous-commande `speed` : trois contextes (un par colonne de la démo, chacun sa dose), poids partagés par mmap, génération entrelacée token par token, 64 tokens par flux, vecteur ×0,5 sur un flux.

| Threads | 1 flux (tokens/s) | 3 flux : par flux / total (tokens/s) | Prompt (22 tokens, 3 flux) |
|---|---|---|---|
| 4 | 14,7 | 4,9 / 14,8 | 1,2 s |
| 8 | 14,2 | 4,8 / 14,3 | 1,1 s |

`llama-bench` b12118 `-ngl 0`, tg64 : 14,4 t/s à 4 threads, 14,9 à 8 (le pp128 de ce binaire, ~445 t/s, passe par le GPU même à `-ngl 0` et n'est pas retenu).

- **Threads limités à 4** : le nombre de threads du portable n'est pas connu ici ; 4 est l'hypothèse basse d'un portable courant. 8 ne fait rien gagner : la génération est limitée par la bande passante mémoire (2,5 Go de poids relus par token), pas par le calcul.
- **Estimation pour trois flux : ~5 tokens/s par colonne sur la tour** (≈ 15 tokens/s au total). Le portable en aura moins s'il a moins de bande passante mémoire (DDR4 ou LPDDR bicanal : ordre de grandeur 2 à 4 tokens/s par colonne, extrapolation non mesurée). La mesure sur le portable se fait en 4.3. Repli si c'est trop lent : servir le 2B en démo (2 fois moins de poids).

## Incertitudes et suites

- Trois prompts et une génération par dose : les plages sont des ordres de grandeur, à resserrer par le banc (epic 3) sur plus d'items.
- Score lexical seulement : pas de juge, pas d'exactitude ; le refus de la tâche à forte dose négative (4B) n'est vu qu'à la lecture.
- Gemma 3 et Phi-4-mini non testés (deux candidats par taille au plus).
- `create_completion` échoue sur Qwen3.5 à forte dose (voir Méthode) : le Steerer (2.2) doit utiliser une boucle `sample` / `eval` ou traiter cette erreur.
- Mémoire GPU du 2B et du 0.8B non mesurée : leurs fichiers (1,3 et 0,5 Go) sont plus petits que celui du 4B, mesuré à 3,3 Go.

## Reproduire

```
set LLAMA_CVECTOR_GENERATOR=...\llama-cvector-generator.exe
uv sync --extra llama
uv run python spikes/calibration_scan.py vector models/gguf/Qwen3.5-4B-Q4_K_M.gguf data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf --threads 10
uv run python spikes/calibration_scan.py scan models/gguf/Qwen3.5-4B-Q4_K_M.gguf data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf data/runs/calibration/Qwen3.5-4B-Q4_K_M.jsonl --doses 0,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0,1.5,-0.02,-0.03,-0.05,-0.07,-0.1,-0.15,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-1.5 --layers all,mid,high --threads 6
uv run python spikes/calibration_scan.py summary data/runs/calibration/Qwen3.5-4B-Q4_K_M.jsonl
uv run python spikes/calibration_scan.py speed models/gguf/Qwen3.5-4B-Q4_K_M.gguf data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf --streams 3 --threads 4 --dose 0.5
```

Le scan reprend là où il s'est arrêté (les cellules déjà écrites sont sautées). Le 0.8B n'a pas été balayé au-delà de ±0,3 : il dégénère dès ×0,2.
