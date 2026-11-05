# Pic 1.4 : calibration joie / tristesse, choix des modèles

Statut : balayage réalisé le 2026-11-05 sur la tour (Ryzen 5 7600, 6 cœurs / 12 threads, 32 Go, RTX 3050 8 Go ; balayage sur CPU), sorties re-notées le même jour par le classifieur du banc (revue de la story). Script : `spikes/calibration_scan.py`. Paires : `src/mood_steering/pairs/joy_sadness.jsonl` (100 paires rédigées, format dans `pairs/README.md`). Test : `tests/test_calibration_scan.py` (format des paires, logique pure du balayage et de la décision, sans modèle).

Registre : le vecteur fait exprimer une humeur au texte du modèle (ton joyeux ou triste), il ne lui prête aucun ressenti.

## Défaut `llama_decode` : `create_completion` ignore `max_tokens` sur un texte dégénéré

Reproduit le 2026-11-05 (llama-cpp-python 0.3.36, roue CPU, Qwen3.5-0.8B Q4_K_M, n_ctx 512) :

- Chemin de `LlamaCppBackend` (main) : `reset()`, `llama_set_adapter_cvec` à ×0,2, puis `create_chat_completion(max_tokens=60, temperature=0)`. Sur la plage `mid` (couches 6–18) : 24 générations sur 24 passent (doses 0, ±0,2, ±0,3, ±0,5). Sur toutes les couches (1–23, la plage par défaut du `Steerer`, `il_start=1`) : **6 échecs sur 6 à ×+0,2**, `RuntimeError: llama_decode returned 1` (« failed to find a memory slot for batch of size 1 »). **`reset()` ne l'évite pas** : même résultat avec ou sans.
- Cause trouvée : au moment de l'échec, `n_tokens` vaut exactement `n_ctx` (512, et 1024 avec n_ctx 1024). Le texte dégénéré est une suite de jetons d'octets UTF-8 incomplets (« � ») ; dans `create_completion`, la branche qui retient les octets incomplets fait `continue` **avant** le test `len(completion_tokens) >= max_tokens` (`llama_cpp/llama.py`, l. ~1428 et ~1582). La génération continue donc jusqu'à remplir le contexte, puis `llama_decode` échoue faute de place. Ce n'est ni la mémoire hybride de Qwen3.5 ni le vecteur en soi : le vecteur ne fait que produire le texte dégénéré.
- La boucle explicite `sample` / `eval` du script compte ses jetons et n'échoue pas (3 sur 3 aux mêmes réglages). `create_completion(..., stopping_criteria=…)` qui arrête après 60 jetons générés passe aussi (3 sur 3, `n_tokens` 82 à 89) ; `create_chat_completion` n'accepte pas `stopping_criteria`.
- Conséquence : avec le `Steerer` actuel, une dose qui fait dégénérer le texte (au-delà de `max_scale`, ou sur une plage de couches trop large) ne s'arrête pas à `max_tokens` : elle remplit tout le contexte (2048 jetons par défaut, donc lent) puis lève `RuntimeError`.
- Remèdes (les deux sont nécessaires) : (1) la story 2.3, en cours, remplace `create_chat_completion` par une boucle bas niveau (`sample` / `eval`) bornée à `max_tokens`, comme celle du script ; (2) **toujours** passer au `Steerer` la plage de couches retenue (`il_start` / `il_end`, voir Décision), jamais son défaut « toutes les couches », qui produit la panne reproduite 6 fois sur 6.
- **Piège du tampon (défaut antérieur à la story, relevé en revue)** : `llama_set_adapter_cvec` lit le tampon par couche absolue (couche `il` à l'offset `n_embd × (il − 1)`, `llama-adapter.cpp`, `llama_adapter_cvec::apply`), quelle que soit la plage. Or le `Steerer` exige un vecteur d'exactement `il_end − il_start + 1` lignes et l'envoie tel quel : avec `il_start=6`, `il_end=18`, les 13 lignes vont dans les couches 1 à 13, seules les couches 6 à 18 sont appliquées, d'où un décalage de 5 couches et des couches 14 à 18 sans la bonne direction (vérifié avec un backend d'enregistrement). Le balayage n'est pas touché : le script envoie le tampon complet (couche 1 en tête). Correctif à faire dans `Steerer` / `LlamaCppBackend` avant d'utiliser une plage qui ne commence pas à 1 : envoyer un tampon de `il_end` lignes, à zéro avant `il_start` (ou accepter le vecteur complet et ne passer que la plage).

## Décision

**Option 1, décidée par l'orchestrateur le 2026-11-05.** On garde les trois Qwen3.5 (4B cœur, 2B, 0.8B), plage de couches `mid` (4B 8–24, 2B 6–18, 0.8B 6–18), `max_scale` fixé par le pôle joie : **4B 0,7 ; 2B 0,5 ; 0.8B 0,15**. Raison : une seule famille permet de tester H2 (effet de taille) sans confondre taille et architecture ; aucun autre candidat ne donne trois tailles dans une même famille, et le seul modèle qui passe la règle sur les deux pôles (Llama-3.2-1B) casserait cette comparaison.

- Ces `max_scale` ne sortent pas de la règle `decide` (qui ne donne aucun `max_scale` aux Qwen, voir le tableau) : c'est la borne haute de la plage joie mesurée, choisie par décision.
- **Pôle tristesse non validé par le pic** (refus d'assistant et texte dégénéré). Il sera mesuré par le banc 3.5 : classifieur, 100 items, doses négatives −0,3, −0,5 et −0,7 rapportées à `max_scale`. **H1 peut échouer sur le pôle négatif** ; ce serait un résultat publiable, à rapporter tel quel.
- **Remplaçant documenté : Llama-3.2-1B-Instruct** (seul candidat qui passe la règle, `max_scale` 0,5, couches 4–12, licence Llama 3.2 Community), si le 2B ou le 0.8B se révèle inutilisable.
- La plage de couches est **toujours** passée au `Steerer` (`il_start` / `il_end`) ; voir le défaut `llama_decode` et le piège du tampon en tête.

### Résultat recalculé

Les plages viennent maintenant d'une commande reproductible, `summary --score tone_clf` (voir Reproduire), qui applique la règle écrite (`decide`) sans relecture à la main. Score : classifieur du banc (`mood_steering.eval.tone`, P(joie) − P(tristesse)), refus d'assistant exclus. Le score lexical du premier passage reste en colonne de comparaison.

| Modèle (couches `mid`) | Joie (classifieur) | Tristesse (classifieur) | `max_scale` (règle) | Lexical, refus exclus : joie / tristesse → `max_scale` | Première note (lexical, lecture à la main) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B (6–18 sur 24) | 0,1–0,15 | aucune | **aucun** | 0,05–0,15 / 0,05–0,1 → 0,1 | 0,05–0,15 / 0,05–0,1, 0,1 |
| Qwen3.5-2B (6–18 sur 24) | 0,3–0,5 | 0,15–0,2 ✗ | **aucun** | 0,2–0,5 / 0,05–0,3 → 0,3 | 0,2–0,5 / 0,1–0,4, 0,4 |
| Qwen3.5-4B (8–24 sur 32) | 0,3–0,7 | 0,7–1,0 ✗ | **aucun** | 0,2–0,7 / 0,05–1,0 → 0,7 | 0,2–0,7 / 0,3–0,7, 0,7 |
| Llama-3.2-1B (4–12 sur 16) | 0,15–0,7 | 0,3–0,5 | **0,5** | 0,3–0,7 / 0,3–0,7 → 0,7 | 0,3–0,7 / 0,3–0,7 |
| Llama-3.2-3B (7–21 sur 28) | 0,7–1,0 ✗ | aucune | **aucun** | 0,05–1,5 / aucune → aucun | 0,1–1,5 / aucune |

✗ : max < 1,5 × min (critère de l'epic). Règle (`decide`) : plage `mid` ; `max_scale` = la plus petite des deux bornes hautes (la plus forte dose où les deux pôles restent utiles), seulement si les deux pôles passent le critère 1,5 ×.

**Ce qui change** : avec le classifieur et les refus exclus, **aucun Qwen3.5 ne passe la règle sur le pôle tristesse**, et aucun candidat 3-4B ne la passe sur les deux pôles. Le pôle joie tient (4B : 0,3–0,7, confirmé sur les prompts tenus à l'écart, voir plus bas). Le pôle tristesse du 4B reposait en partie sur des refus : dès ×−0,07, « Raconte ta matinée » reçoit une mise au point d'assistant (« Je ne peux pas raconter ma matinée car je suis une intelligence artificielle… »), que le lexique comptait triste ; et le classifieur ne voit qu'un ton neutre dans le message de report à toutes les doses négatives. Seul Llama-3.2-1B passe les deux pôles (`max_scale` 0,5).

Options qui étaient soumises à l'orchestrateur (la règle écrite ne donne pas trois modèles ; l'option 1 est retenue, voir plus haut) :

1. Garder la famille Qwen3.5 (une seule architecture pour H2, Apache-2.0, sorties françaises), avec un `max_scale` fixé par le pôle joie (4B : 0,7 ; 2B : 0,5 ; 0.8B : 0,15) et le pôle tristesse déclaré **non validé** par ce pic, à mesurer par le banc (H1) ; ou
2. prendre Llama-3.2-1B pour le 1-2B (seul candidat qui passe la règle, `max_scale` 0,5, licence Llama 3.2 Community) et garder Qwen3.5-4B et 0.8B sous l'option 1 ; ou
3. relancer un balayage plus large du pôle tristesse (plus de prompts, prompts narratifs, plage de couches plus étroite) avant de choisir.

Le reste de cette note (couches, quantification, VRAM, vitesse) décrit l'option retenue.

| Rôle | Modèle | Dépôt HF, fichier | Quantif. | Licence | Mémoire GPU mesurée (n_ctx 4096) | Couches | `max_scale` |
|---|---|---|---|---|---|---|---|
| Cœur (3-4B) | Qwen3.5-4B | `unsloth/Qwen3.5-4B-GGUF`, `Qwen3.5-4B-Q4_K_M.gguf` | Q4_K_M | Apache-2.0 | 3 387 Mio | 8–24 | 0,7 |
| 1-2B | Qwen3.5-2B | `unsloth/Qwen3.5-2B-GGUF`, `Qwen3.5-2B-Q4_K_M.gguf` | Q4_K_M | Apache-2.0 | 1 875 Mio | 6–18 | 0,5 |
| Petit | Qwen3.5-0.8B | `unsloth/Qwen3.5-0.8B-GGUF`, `Qwen3.5-0.8B-Q4_K_M.gguf` | Q4_K_M | Apache-2.0 | 1 153 Mio | 6–18 | 0,15 |

Doses en multiplicateur du vecteur (norme 1 par couche, voir Normalisation). Pour le Steerer : tableau passé au binding = d × `max_scale` × vecteur, d dans [−1, 1], sur la plage de couches `mid` (à passer explicitement, voir le défaut `llama_decode` et le piège du tampon en tête). Remplaçant documenté du 1-2B ou du petit : Llama-3.2-1B-Instruct (Q4_K_M, couches 4–12, `max_scale` 0,5).

Raisons qui tiennent :

- **Plage de couches « mid »** (de 25 % à 75 % de la profondeur) : avec les deux scores, c'est la plage qui donne la plus large plage joie sur les trois Qwen. `all` dégénère plus tôt (4B : joie 0,3–0,5 au classifieur ; 0.8B : joie 0,03 seulement) ; `high` (50 % à 85 %) demande des doses plus fortes (4B : 1,0–1,5 ; 2B : 0,7–1,0, critère non atteint) et ne donne aucun pôle tristesse sur le 4B et le 0.8B.
- **Qwen3.5-4B pour le cœur** : pôle joie le plus large des 3-4B (0,3–0,7, 2,3 ×), confirmé sur des prompts tenus à l'écart ; tient en Q4 dans 8 Go (3,4 Gio mesurés) ; Apache-2.0. Llama-3.2-3B-Instruct ne montre aucun pôle tristesse et refuse déjà, sans vecteur, deux des trois tâches de jeu de rôle.
- **Qwen3.5-0.8B** : imposé par l'epic. **Fragile** : pôle joie 0,1–0,15 au classifieur, soit un rapport de 1,5 tout juste égal au seuil (la règle ne l'accepte que grâce à la tolérance flottante) ; au score lexical, le ton passe de 0,00 à ×−0,03 à −0,58 à ×−0,05, un saut sur un seul cran de la grille ; le texte dégénère dès ×0,2. Toute plage de ce modèle est à resserrer par le banc.
- **Une seule famille** : l'effet de taille (H2) ne mélange pas taille et famille.

## Méthode

- **Vecteur** : `llama-cvector-generator` b12118 (CPU), `--method mean`, 100 paires, un vecteur par modèle dans `data/moods/{model_id}/joy_sadness.gguf` (non versionné ; `model_id` = nom du fichier GGUF sans extension). Le script passe par `mood_steering.calibration.calibrate` (même échappement des invites, même variable `MOOD_CVECTOR_GENERATOR` que la bibliothèque). `mean` plutôt que `pca` : la différence moyenne positif − négatif a un signe fixé par construction, alors que `pca` (itération de puissance partant d'un vecteur aléatoire initialisé sur `time(0)`) n'est ni déterministe ni orienté. Génération : 4 à 7 min par modèle sur 10 threads. Les vecteurs du balayage ont été produits avec la première version des paires ; six paires ont depuis été reformulées (tournures qui déclaraient un ressenti, « Je me sens… »), un vecteur régénéré différera donc un peu.
- **Normalisation** : le générateur normalise déjà chaque direction `mean` à une norme L2 de 1 par couche (`tools/cvector-generator/mean.hpp`, relu : normes 1,000 sur toutes les couches ; note du pic 1.3 corrigée). Choix retenu : **garder la norme unitaire par couche, sans autre normalisation** ; l'écart d'échelle entre modèles est absorbé par `max_scale`, propre à chaque modèle.
- **Balayage** : trois plages de couches (`all`, `mid`, `high`, en fractions de la profondeur, `layer_range`), doses 0, ±0,02 à ±1,5, trois questions françaises neutres en tour de chat (matinée, message de report de réunion, trajet gare – centre-ville), 60 tokens max, glouton (`top_k=1`, `repeat_penalty=1.1`), une génération par cellule, CPU. Avant chaque génération, `reset()` du contexte. Vecteur posé par `llama_set_adapter_cvec(ctx, tampon, longueur, n_embd, il_start, il_end)` ; dose 0 = retrait (`None`). Sorties : `data/runs/calibration/{model_id}.jsonl` (non versionné), re-notées dans `{model_id}.scored.jsonl`.
- **Génération** : boucle explicite `sample` / `eval`, qui compte ses jetons (voir le défaut `llama_decode` en tête). Une génération en erreur reste dans le JSONL avec son message, mais `aggregate` l'exclut de la moyenne et de la part cohérente et la compte à part (colonne `erreurs` de `summary` ; aucune dans ce balayage).
- **Score de ton principal** (`rescore`, champ `tone_clf`) : P(joie) − P(tristesse) de `tabularisai/multilingual-emotion-classification`, révision épinglée dans `mood_steering.eval.tone` ; le même score que le banc. **Score lexical** (`tone_score`, champ `tone`) : gardé pour comparaison ; il compte les mots de politesse et de refus (« désolé », « je ne peux pas… ») comme tristesse.
- **Refus d'assistant** (`is_refusal`, motifs `REFUSAL_PATTERNS` du script) : dans les 160 premiers caractères de la réponse, « en tant qu'IA », « je suis une intelligence artificielle / un modèle de langage / un assistant », « je ne suis pas un être vivant / humain », « je ne peux pas écrire / fournir / raconter / répondre… », « je ne suis pas capable de ressentir / vivre », « je n'ai pas de corps / de matinée / d'émotions… », « je ne possède pas », « pas de corps physique », et leurs équivalents anglais. Au-delà de 160 caractères, la phrase appartient en général au jeu de rôle (« je ne peux pas fournir les informations » dans le message de report) : contrôlé à la main sur les 158 sorties signalées du balayage. Une sortie refusée compte comme non cohérente et sort de la moyenne du ton.
- **Cohérence** (`is_coherent`, `is_french`) : au moins 8 mots, part de mots distincts ≥ 0,45, ≥ 70 % de lettres, aucun trigramme répété 3 fois, plus de mots-outils français qu'anglais.
- **Plage utile** (`useful_range`) : plus longue suite contiguë de doses d'un signe dont au moins 2 sorties sur 3 sont cohérentes, en français et sans refus, et dont le ton moyen s'écarte de celui de la dose 0 d'au moins 0,25 dans le bon sens (seuils comparés avec une tolérance de 1e-9 : 0,42 − 0,17 vaut 0,2499999… en flottant). Critère de l'epic : max ≥ 1,5 × min (`meets_ratio`). Décision : `decide` (voir plus haut).
- **Limites** : trois prompts et une génération par dose pour la sélection ; un seul prompt qui bascule fait bouger la moyenne d'un tiers. Les seuils (0,25, 2/3) ont été fixés pour le score lexical et repris tels quels pour le classifieur, sans réglage après coup.

## Validation sur des prompts tenus à l'écart (Qwen3.5-4B)

Cinq autres questions neutres (`HOLDOUT_PROMPTS` : week-end, mot aux voisins, vue d'un bureau, trajet en bus, invitation à dîner), plage `mid`, doses 0, ±0,2 à ±1,0, même protocole (65 générations, ~5 min de génération sur CPU). `summary --score tone_clf` :

| Dose | +0,2 | +0,3 | +0,4 | +0,5 | +0,7 | +1,0 | −0,2 | −0,3 | −0,4 | −0,5 | −0,7 | −1,0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ton (classifieur) | +0,01 | +0,39 | +0,52 | +0,63 | +0,89 | +0,22 | −0,03 | −0,12 | −0,25 | −0,38 | −0,17 | −0,48 |
| Part utile | 0,8 | 1 | 1 | 1 | 1 | 0,6 | 0,8 | 0,6 | 0,6 | 0,4 | 0,6 | 0,4 |
| Refus | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 2 | 2 | 0 |

Base (dose 0) : ton 0,00, un refus sur cinq. Pôle joie : **0,3–0,7, identique à la sélection**. Pôle tristesse : aucune plage utile ; le ton descend (−0,25 à −0,48), mais au plus 3 sorties sur 5 restent utilisables, le reste étant des refus ou du texte dégénéré. Le constat de la sélection tient donc hors des prompts qui l'ont produit.

## Mémoire GPU (mesurée)

`llama-completion` b12118 CUDA 12.4 (`models/bin/llama-b12118-cuda12.4/`), `-ngl 99 -c 4096 -n 200`, un contexte, pic de `nvidia-smi --query-gpu=memory.used` relevé toutes les 0,5 s pendant l'exécution (0 Mio avant) : **Qwen3.5-4B 3 387 Mio**, Qwen3.5-2B 1 875 Mio, Qwen3.5-0.8B 1 153 Mio, sur 8 192 Mio. `llama-bench -ngl 99` (n_ctx 512) sur le 4B : pic 3 275 Mio, pp512 2 517 t/s, tg128 60,6 t/s. Trois contextes de 4096 (démo trois colonnes sur un seul modèle) ajoutent chacun leur cache, non mesuré ici.

## Vitesse CPU du 4B

Qwen3.5-4B Q4_K_M, GPU désactivé (roue CPU de llama-cpp-python, `n_gpu_layers=0`), sous-commande `speed` : trois instances `Llama` distinctes (une par colonne, donc trois chargements du modèle ; le partage des pages par mmap n'est pas vérifié et le réarrangement CPU des poids peut en faire des copies), génération entrelacée token par token, 64 tokens par flux, vecteur ×0,5 sur un flux.

| Threads | 1 flux (tokens/s) | 3 flux : par flux / total (tokens/s) | Prompt (22 tokens, 3 flux) |
|---|---|---|---|
| 4 | 14,7 | 4,9 / 14,8 | 1,2 s |
| 8 | 14,2 | 4,8 / 14,3 | 1,1 s |

`llama-bench` b12118 `-ngl 0`, tg64 : 14,4 t/s à 4 threads, 14,9 à 8.

- **4 threads est une hypothèse** : le nombre de threads du portable n'est pas connu ici ; 4 est l'hypothèse basse d'un portable courant. 8 ne fait rien gagner : la génération est limitée par la bande passante mémoire, pas par le calcul.
- **Estimation pour trois flux : ~5 tokens/s par colonne sur la tour** (≈ 15 tokens/s au total). Le portable en aura moins s'il a moins de bande passante mémoire (extrapolation non mesurée). **La mesure sur le portable se fait en 4.3.** Repli si c'est trop lent : servir le 2B en démo.

## Incertitudes et suites

- Pôle tristesse non validé sur les Qwen3.5 (voir Décision) ; le banc 3.5 le mesurera (100 items, −0,3 / −0,5 / −0,7), H1 peut y échouer.
- Le 0.8B est au seuil du critère : plage indicative seulement.
- Gemma 3 et Phi-4-mini non testés (deux candidats par taille au plus).
- `create_completion` / `create_chat_completion` et texte dégénéré : voir le défaut `llama_decode` en tête (story 2.3) ; décalage du tampon du `Steerer` pour une plage qui ne commence pas à 1 : à corriger avant tout usage de la plage `mid`.

## Reproduire

```
set MOOD_CVECTOR_GENERATOR=...\llama-cvector-generator.exe
uv sync --extra llama --extra eval
uv run python spikes/calibration_scan.py vector models/gguf/Qwen3.5-4B-Q4_K_M.gguf --threads 10
uv run python spikes/calibration_scan.py scan models/gguf/Qwen3.5-4B-Q4_K_M.gguf data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf data/runs/calibration/Qwen3.5-4B-Q4_K_M.jsonl --doses 0,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0,1.5,-0.02,-0.03,-0.05,-0.07,-0.1,-0.15,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0,-1.5 --layers all,mid,high --threads 6
uv run python spikes/calibration_scan.py rescore data/runs/calibration/Qwen3.5-4B-Q4_K_M.jsonl data/runs/calibration/Qwen3.5-4B-Q4_K_M.scored.jsonl
uv run python spikes/calibration_scan.py summary data/runs/calibration/Qwen3.5-4B-Q4_K_M.scored.jsonl --score tone_clf
uv run python spikes/calibration_scan.py scan models/gguf/Qwen3.5-4B-Q4_K_M.gguf data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf data/runs/calibration/Qwen3.5-4B-Q4_K_M.holdout.jsonl --doses 0,0.2,0.3,0.4,0.5,0.7,1.0,-0.2,-0.3,-0.4,-0.5,-0.7,-1.0 --layers mid --prompts holdout --threads 10
uv run python spikes/calibration_scan.py speed models/gguf/Qwen3.5-4B-Q4_K_M.gguf data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf --streams 3 --threads 4 --dose 0.5
```

`summary` sans `--score` donne la colonne lexicale. La dernière ligne de `summary` (`DÉCISION`) est la décision par la règle. Le scan reprend là où il s'est arrêté (les cellules déjà écrites sont sautées). Le 0.8B n'a pas été balayé au-delà de ±0,3 : il dégénère dès ×0,2.

## Revue

### Passe 1 (2026-11-05, SHA revu cdd8229)

- Important, high, second avis : score lexical qui compte refus et dénégations comme tristesse. Corrigé : re-notation des sorties existantes par le classifieur du banc (`rescore`), détecteur de refus, plages et `max_scale` recalculés par `decide` ; la décision change (pôle tristesse des Qwen non validé), remontée à l'orchestrateur.
- Important, medium, second avis : plage retenue à la main contre `summary`. Corrigé : `decide` applique la règle, la note cite la commande.
- Important, medium, second avis : fragilité du 0.8B non dite. Corrigé (Raisons).
- Important, medium, second avis : sélection et contrôle sur les mêmes prompts. Corrigé : validation du 4B sur 5 prompts tenus à l'écart (joie confirmée, tristesse non utile).
- Important, medium, second avis : VRAM « prévue » et non mesurée. Corrigé : mesurée (`nvidia-smi`, 3 387 Mio pour le 4B à n_ctx 4096).
- Important, medium, second avis : description de `speed` (trois `Llama`, pas des poids partagés prouvés) et 4 threads présentés comme mesure. Corrigé.
- Important, medium, second avis : échec de `create_completion` non expliqué. Corrigé : reproduit, cause trouvée (`max_tokens` ignoré sur octets UTF-8 incomplets), `reset()` ne suffit pas ; risque en tête de note, remonté à l'orchestrateur.
- Important, medium, second avis : une génération en erreur comptée comme incohérente. Corrigé : exclue de `aggregate`, comptée à part.
- Nit, low, second avis : « non normalisé » dans la note du pic 1.3. Corrigé.
- Nit, low, second avis : six paires qui déclaraient un ressenti. Corrigé (100 paires, sans doublon).
- Nit, low, second avis : `MIN_SHIFT` au seuil flottant, `max_scale` du 2B plus strict que la règle. Corrigé (tolérance `EPS`, `max_scale` calculé).
- Nit, low, second avis : deux variables d'environnement et deux échappements. Corrigé : `vector` passe par `calibrate` et `MOOD_CVECTOR_GENERATOR`.
- Complexité : 0 constat (structures bornées par la grille de doses).

### Passe 2 (2026-11-05, SHA revu ad1e672, main fusionnée par merge)

- Important, medium, orchestrateur : note encore rédigée comme « décision à trancher ». Corrigé : section Décision en tête (option 1 et sa raison), `max_scale` et couches par modèle dans le tableau, pôle tristesse renvoyé au banc 3.5 (H1 peut échouer), Llama-3.2-1B remplaçant, défaut `llama_decode` avec cause et les deux remèdes (boucle bornée de la story 2.3, plage toujours passée au `Steerer`).
- Pre-existing, high, revue : `Steerer` envoie un vecteur de `il_end − il_start + 1` lignes alors que `llama_adapter_cvec::apply` lit le tampon par couche absolue ; avec `il_start=6`, direction décalée de 5 couches. Reporté (hors story, `steering.py` touché par la story 2.3) : documenté dans la note, à corriger avant tout usage de la plage `mid`.
- Vérifié : `summary --score tone_clf` sur les cinq `*.scored.jsonl` et le holdout redonne exactement les plages et décisions du tableau ; `rescore` non relancé (téléchargement du classifieur) ; `is_refusal` testé sur sorties réelles, fenêtre de 160 caractères comprise.
- Complexité : 0 constat.
