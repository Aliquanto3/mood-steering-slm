# Pic 1.3 : relire le GGUF de `llama-cvector-generator` en flottants

Statut : réalisé le 2026-11-04 sur la tour (RTX 3050 8 Go, CPU pour ce pic). Script : `spikes/read_cvector_gguf.py`. Test : `tests/test_read_cvector_gguf.py` (GGUF synthétique écrit par `gguf` avec le nommage réel, aucun modèle).

## Résultat

- **Lecture : OK.** La bibliothèque `gguf` relit le vecteur, chaque couche rendue en `float32` de la dimension cachée (1024 pour Qwen3.5-0.8B).
- **Équivalence : OK à l'échelle ×1.** Même graine (42), même prompt, température 0, 40 tokens : la sortie de `llama_set_adapter_cvec` (llama-cpp-python) est identique, caractère pour caractère, à celle de `llama-completion --control-vector`. Le repli « vecteur maison » n'est donc pas nécessaire pour la lecture.
- Limite de la preuve : à ×1 le vecteur moyen brut écrase le texte (sortie dégénérée, `締!<!<…`), ce qui prouve que le vecteur est appliqué de la même façon mais pas la finesse numérique. À ×0,15 et ×0,3, les deux chemins montrent le même comportement (« Let! … ») avec un écart de quelques tokens. Les sorties sans vecteur divergent déjà entre CLI et binding (voir Versions) : c'est une dérive de build (noyaux Q4 CPU, Qwen3.5 hybride), pas une erreur de lecture.
- Conséquence pour la story 1.4 : le vecteur `--method mean` n'est pas normalisé ; la plage de doses utile est très inférieure à 1 sur ce modèle (bien en dessous de 0,15 pour garder un texte lisible). À balayer finement.

## Nommage des tenseurs (constaté et lu dans les sources b12118)

Fichier de `export_gguf` (`tools/cvector-generator`) :

- architecture GGUF `controlvector`, métadonnées `controlvector.model_hint` (ici `qwen35`) et `controlvector.layer_count` (entier ; ici 23).
- un tenseur par couche, nommé **`direction.N`**, `N` de 1 à `layer_count`, `float32` (`F32`), 1D, forme `[n_embd]`.
- **Couches absentes** : la couche 0 n'existe jamais (le chargeur la refuse : « invalid (zero) direction tensor layer index »). Qwen3.5-0.8B a 24 couches : le fichier généré contient `direction.1` à `direction.23`, la couche 24 est absente (`layer_count` = 23, soit n_couches − 1). Un fichier peut aussi omettre des couches intermédiaires : elles valent zéro.
- Le chargeur de la CLI refuse : nom hors `direction.N`, `N` = 0 ou > nombre de couches, type non F32, forme non 1D, dimensions inégales. `read_cvector` refuse les mêmes cas.
- Tampon pour `llama_set_adapter_cvec` : `n_couches × n_embd` flottants, **couche 1 en tête** (offset `n_embd × (il − 1)`), couches absentes à zéro ; appel `(ctx, données, longueur, n_embd, 1, n_couches)`. `CVector.flat()` le construit.
- Particularité Windows : `--control-vector-scaled FICHIER:ÉCHELLE` coupe au premier `:`, donc échoue avec un chemin `C:/…`. Le script passe un nom relatif et fixe le dossier de travail.

## Versions et build

| Élément | Version |
|---|---|
| Générateur | `llama-cvector-generator` compilé depuis les sources llama.cpp b12118, build CPU (zig), `--method mean` |
| CLI de comparaison | `llama-completion` des binaires officiels b12118 CUDA 12.4, lancé avec `-ngl 0` |
| Binding | `llama-cpp-python` 0.3.36, roue précompilée CPU de l'index abetlen (`whl/cpu`, extra `llama`) ; expose `llama_set_adapter_cvec` |
| Bibliothèque `gguf` | résolue par `uv add gguf` (voir `uv.lock`) |
| Modèle | Qwen3.5-0.8B Q4_K_M (unsloth), `models/gguf/`, non versionné |

Le générateur n'existe pas dans les builds publiés : il faut le compiler (CMake ou zig, sans compilateur MSVC ici). Six paires joie / tristesse rédigées dans le script suffisent à produire un vecteur (génération : 16 s sur CPU). Appeler la DLL b12118 depuis le binding 0.3.36 (`LLAMA_CPP_LIB_PATH`) échoue au chargement du modèle (ABI différente) : les deux chemins ne partagent donc pas le même build, d'où la dérive sans vecteur.

## Repli : vecteur maison

Non déclenché (lecture et équivalence réussies). Évaluation tout de même : un vecteur maison (moyenne par couche des différences d'activations entre paires, captées via llama-cpp-python avec `embeddings`/`cb_eval`) est faisable, sans le générateur, mais coûte du code à écrire et à tester : capture des états cachés par couche (accroche `cb_eval` sur `l_out`, absente de l'API haut niveau, à passer par ctypes), alignement des positions de tokens entre paires, sérialisation au même format. Ordre de grandeur : une à deux journées de pic, contre zéro pour le générateur. Il resterait utile si le générateur devenait impossible à construire sur une machine cible (compilation de llama.cpp). La mise en cache `data/moods/` (story 2.1) peut s'appuyer sur le générateur, compilé une fois.

## Reproduire

```
set LLAMA_CVECTOR_GENERATOR=...\llama-cvector-generator.exe
set LLAMA_COMPLETION=...\llama-completion.exe
uv sync --extra llama
uv run python spikes/read_cvector_gguf.py generate models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf data/moods/pic13-joie.gguf
uv run python spikes/read_cvector_gguf.py show data/moods/pic13-joie.gguf
uv run python spikes/read_cvector_gguf.py equiv models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf data/moods/pic13-joie.gguf --scale 1.0
```

Registre : ces vecteurs servent à faire exprimer une humeur au modèle (ton du texte), jamais à lui prêter un ressenti.
