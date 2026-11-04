# Pic 1.2 : vecteur de contrôle réappliqué entre deux tokens

Verdict : **GO pour CAP-3** (variation de la dose en cours de génération, cache KV conservé).

## Protocole

`spikes/cvec_between_tokens.py` ; génération gloutonne token par token (`llm.sample(top_k=1, temp=0)`, `llm.eval([tok])`), CPU, prompt `The weather this morning was`, 60 tokens, N = 20.
Modèle : Qwen3.5-0.8B Q4_K_M (`unsloth/Qwen3.5-0.8B-GGUF`, dans `models/`, non versionné). Vecteur synthétique : direction gaussienne normalisée par couche, norme 8, couches 1 à 23, graine 0. Quatre suites : `base` (rien), `full` (vecteur dès le début), `late` (vecteur posé avant le token 20), `off` (vecteur dès le début, retiré avant le token 20). Contrôle : `base` relancée est identique (déterminisme : oui).

## Résultats (indices dans la suite générée, 0 = premier token)

| Comparaison | Premier token différent |
|---|---|
| `late` vs `base` | 21 |
| `full` vs `base` | 1 |
| `off` vs `full` | 21 |
| `off` vs `base` | 1 |

- Avant N, `late` est identique à `base` : le vecteur ne fuit pas en amont. La divergence apparaît au token N+1 et non N : le token N est échantillonné à partir des logits de l'évaluation précédente ; le vecteur n'agit que sur la prochaine évaluation (`llama_decode`). À retenir pour le calendrier de dose : une dose changée « avant le token N » agit sur le token N+1.
- Le cache KV est conservé (aucun `reset` ni ré-évaluation du prompt) et le vecteur agit quand même : il s'applique aux nouveaux tokens à l'évaluation, pas au cache.
- `late` : `very cold.\nThe weather this morning was very cold.\nThe weather this morning was very cold.` puis `TrailTrailTrail…`. La suite bascule d'un coup en sortie dégénérée (la norme 8 est volontairement excessive pour juger la divergence ; la calibration relève de 1.4).

## Retrait en cours de génération (dose 0)

L'effet **cesse en grande partie mais ne s'efface pas** : `off` diffère de `full` dès le token 21 (les `inking…Trail…` s'arrêtent) et la suite redevient cohérente, mais elle reste différente de `base` (`... was good.` au lieu de `... was very cold.`, et le début `very inking…Trail` reste dans le contexte). Deux causes cumulées : le texte déjà généré sous vecteur reste dans le contexte, et les K/V des positions évaluées sous vecteur gardent la trace du vecteur. Pour revenir à une suite strictement neutre, il faut repartir d'un contexte propre (réévaluer le texte sans vecteur ou `reset`). Conséquence pour CAP-3 : une courbe de dose par token est faisable, mais l'effet d'une dose forte a une rémanence.

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

- Vecteur synthétique : l'effet d'un vecteur de contraste réel (1.3, 1.4) n'est pas testé ; la rémanence doit être remesurée à dose utile (plus faible).
- Seul le CPU est testé (roue sans CUDA) ; le chemin GPU n'est pas vérifié ici.
- Un seul modèle (Qwen3.5-0.8B, architecture hybride) ; un modèle 3-4B n'est pas testé.
- Install : `uv sync --extra llama` (index `llama-cpu` déclaré dans `pyproject.toml`) ; CPython 3.13 avait une roue disponible. La CI n'installe pas l'extra.
