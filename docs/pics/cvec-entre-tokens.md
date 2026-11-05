# Pic 1.2 : vecteur de contrôle réappliqué entre deux tokens

Verdict : **GO pour CAP-3** (variation de la dose en cours de génération, cache KV conservé), établi à une dose non dégénérée (norme 0,6 par couche) et confirmé de 0,7 à 8.

## Protocole

`spikes/cvec_between_tokens.py` ; génération gloutonne token par token (`llm.sample(top_k=1, temp=0)`, `llm.eval([tok])`), CPU, prompt `The weather this morning was`, 60 tokens, N = 20.
Modèle : Qwen3.5-0.8B Q4_K_M (`unsloth/Qwen3.5-0.8B-GGUF`, dans `models/`, non versionné). Vecteur synthétique : direction gaussienne normalisée par couche, couches 1 à 23, graine 0, norme balayée (0,5 ; 0,6 ; 0,7 ; 0,8 ; 0,9 ; 1 ; 2 ; 4 ; 8).

Le prompt est toujours évalué **sans** vecteur. Cinq suites par dose :

- `base` : aucun vecteur ;
- `full` : vecteur posé avant le premier token généré (il agit donc à partir du token 1) ;
- `late` : vecteur posé avant le token N ;
- `off` : comme `full`, vecteur retiré avant le token N (dose 0), cache KV conservé ;
- `replay` : contrôle de `off` ; `reset`, puis les N premiers tokens de `off` réévalués **sans** vecteur (préfixe imposé), suite gloutonne sans vecteur. Même texte que `off`, mais K/V propres.

Contrôle de déterminisme : `base` est relancée et le script s'arrête en erreur si elle diffère (elle n'a pas différé).

## Résultats (indices dans la suite générée, 0 = premier token)

| Norme | `full` vs `base` | `late` vs `base` | `off` vs `full` | `off` vs `replay` | Sortie de `full` |
|---|---|---|---|---|---|
| 0,5 | — | — | — | — | identique à `base` |
| 0,6 | 8 | 26 | 26 | 26 | cohérente (« is very cold » au lieu de « was very cold ») |
| 0,7 | 1 | 26 | 24 | 24 | cohérente (« very good. The weather is good. ») |
| 0,8 à 1 | 1 à 4 | 22 à 24 | — | — | « very good. » puis `.
` répété |
| 2 à 8 | 1 | 21 à 22 | 21 | 21 | dégénérée (`anyany…`, `AAAA…`, `Trail…`) |

« — » : suites identiques sur les 60 tokens.

- Avant N, `late` est identique à `base` à toutes les doses : le vecteur ne fuit pas en amont. La divergence apparaît au plus tôt au token N+1 : le token N est échantillonné à partir des logits de l'évaluation précédente ; le vecteur n'agit que sur la prochaine évaluation (`llama_decode`). À retenir pour le calendrier de dose : une dose changée « avant le token N » agit sur le token N+1. À faible dose, la divergence vient plus tard (token 26 à 0,6 et 0,7) : le choix glouton ne bascule qu'au premier token où la marge entre candidats est faible.
- Le cache KV est conservé (aucun `reset` ni réévaluation) et le vecteur agit quand même : il s'applique aux nouveaux tokens à l'évaluation.
- Dose : sous 0,6, aucun effet visible avec ce vecteur aléatoire ; dès 0,8, la sortie se dégrade. La fenêtre utile d'un vecteur aléatoire est donc étroite ; celle d'un vecteur de contraste réel relève de 1.4.

## Retrait en cours de génération (dose 0)

Mesuré :

- Le retrait est effectif : à 0,6, 0,7 et de 2 à 8, `off` quitte `full` juste après N (tokens 21 à 26). À 0,6, `off` reprend même la tournure de `base` (« was very cold ») après trois lignes en « is ».
- `off` ne redevient pas identique à `base` : le texte généré sous vecteur reste dans le contexte.
- Le contrôle `replay` (même texte, K/V réévalués sans vecteur) **diffère de `off`** dès que `off` quitte `full`, à toutes ces doses. Les K/V calculés sous vecteur pèsent donc sur la suite ; mais c'est `replay` qui garde le plus la trace : à 0,6 il reste sur « is », à 2-8 il continue la sortie dégénérée par recopie du contexte, alors que `off` revient à une suite cohérente.
- De 0,8 à 1, `off`, `full` et `replay` sont identiques : la répétition `.
` installée avant N s'entretient d'elle-même, avec ou sans vecteur.

Supposé, non mesuré : le mécanisme exact par lequel les K/V « sous vecteur » réduisent la recopie (une hypothèse : les clés décalées attirent moins l'attention des nouvelles positions propres).

Conséquence pour 2.3 (calendrier de dose) : **retirer le vecteur suffit** à arrêter son effet direct ; réévaluer le contexte sans vecteur n'efface pas la rémanence et, sur ce prompt et ce vecteur aléatoire, la renforce (constat qualitatif sur les textes imprimés, `replay` vs `full` n'est pas chiffré). La rémanence après retrait a deux sources mesurées : le texte déjà produit et les K/V calculés sous vecteur (`off` ≠ `replay`) ; seul un retour à un contexte sans ce texte (`reset` sur le prompt seul) supprime les deux. Le calendrier peut donc changer la dose par appel `llama_set_adapter_cvec` sans réévaluation.

## Signature du binding

llama-cpp-python 0.3.36 (roue précompilée `py3-none-win_amd64` de l'index CPU de abetlen), ggml 0.25.3 commit `0c1e570` (via `ggml_version()` / `ggml_commit()`). Le numéro de build llama.cpp (bNNNN) n'est pas exposé par la roue ; les binaires de comparaison du dépôt sont b12118.

```python
llama_cpp.llama_set_adapter_cvec(
    ctx,        # llama_context_p : llm._ctx.ctx
    data,       # POINTER(c_float) : (c_float * len)(*floats), ou None pour retirer
    len,        # c_size_t : nombre total de flottants dans data
    n_embd,     # c_int32 : dimension cachée (llm.n_embd(), 1024 ici)
    il_start,   # c_int32 : première couche ciblée, incluse
    il_end,     # c_int32 : dernière couche ciblée, incluse
) -> int        # 0 = succès, -1 = échec
```

- Forme : tampon plat de `n_embd` flottants par couche, **la couche 1 en premier** (la couche 0 n'a pas de vecteur). Pour `il_start=1`, `il_end=23` (24 couches, `llama_model_n_layer`) : `1024 × 23 = 23552` flottants. Le tampon est copié par llama.cpp à l'appel.
- Plage : `1 <= il_start <= il_end <= n_layer - 1` en usage normal. Observé : `n_embd` erroné renvoie -1 ; `il_end = n_layer`, `il_start = 0` et une longueur trop courte renvoient 0 sans erreur, donc le binding ne valide pas tout. À valider côté Python.
- Retrait : `llama_set_adapter_cvec(ctx, None, 0, n_embd, 0, 0)` renvoie 0 et désactive le vecteur (équivalent d'une dose 0). Un tampon de zéros donne le même effet pour un coût plus élevé.
- Remplacement : un nouvel appel écrase le précédent ; pas besoin de retirer d'abord.
- Accès au contexte : `Llama._ctx.ctx` est un attribut privé ; à encapsuler dans `steering`.

## Incertitudes et suites

- Vecteur synthétique aléatoire : l'effet d'un vecteur de contraste réel (1.3, 1.4) n'est pas testé ; la rémanence et le contrôle `replay` sont à remesurer à sa dose utile. Un seul prompt et une seule graine.
- Seul le CPU est testé (roue sans CUDA) ; le chemin GPU n'est pas vérifié ici.
- Un seul modèle (Qwen3.5-0.8B, architecture hybride) ; un modèle 3-4B n'est pas testé.
- Install : `uv sync --extra llama` (index `llama-cpu` déclaré dans `pyproject.toml`) ; CPython 3.13 avait une roue disponible. L'index est `explicit` : seul `llama-cpp-python` y est cherché. Le verrou ne contient pas de roue macOS x86_64 (Intel) : l'extra n'y est pas installable sans compilation. La CI n'installe pas l'extra.
