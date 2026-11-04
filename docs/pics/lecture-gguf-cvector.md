# Pic 1.3 : relire le GGUF de `llama-cvector-generator` en flottants

Statut : réalisé le 2026-11-04 sur la tour (RTX 3050 8 Go, CPU pour ce pic). Script : `spikes/read_cvector_gguf.py`. Test : `tests/test_read_cvector_gguf.py` (GGUF synthétique écrit par `gguf` avec le nommage réel, aucun modèle).

## Résultat

- **Lecture : OK.** La bibliothèque `gguf` relit le vecteur, chaque couche rendue en `float32` de la dimension cachée (1024 pour Qwen3.5-0.8B). Sortie de `show` (extrait) :

  ```
  direction.1	forme=[1024]	type=F32
  …
  direction.23	forme=[1024]	type=F32
  n_embd=1024 layer_count=23 model_hint=qwen35
  couches présentes=[1, …, 23] absentes=[]
  ```

- **Équivalence : ce qui est prouvé.** Graine 42, même prompt, 40 tokens, samplers épinglés des deux côtés (glouton : `--temp 0`, `--repeat-penalty 1`, `-c 512` en CLI ; `temperature=0`, `repeat_penalty=1`, `n_ctx=512` côté binding). Avec vecteur, `llama_set_adapter_cvec` (llama-cpp-python 0.3.36) et `llama-completion --control-vector-scaled` (b12118) donnent un texte identique à ×1 et à ×0,3, et les mêmes 90 premiers caractères à ×0,1. Un mauvais décalage de couches ou un mauvais tampon ne donnerait pas ces égalités : la lecture, le mapping `direction.N` → couche et les flottants passés sont les mêmes.
- **Ce qui n'est pas prouvé : l'égalité au token près entre les deux builds.** Sans vecteur, la CLI b12118 et la roue 0.3.36 divergent dès le premier mot (noyaux Q4 CPU différents, Qwen3.5 hybride ; la DLL b12118 ne se charge pas depuis la roue 0.3.36, ABI différente). Aux doses faibles, chaque chemin s'écarte donc de sa propre base avec du bruit de build : ×0,1, même début puis suites différentes ; ×0,05, textes différents mais même sens de changement. Comparaison chacun contre sa propre base (`equiv` l'affiche) :

  | Dose | CLI = binding avec vecteur | Divergence avec sa base (caractère, CLI / binding) | Texte |
  |---|---|---|---|
  | ×1 | oui | 0 / 0 | dégénéré (`締!<!<…`) |
  | ×0,3 | oui | 0 / 0 | dégénéré (« Let! ») |
  | ×0,15 | non (retour à la ligne près) | 0 / 0 | répétitions « Let's let! » |
  | ×0,1 | 90 premiers caractères | 0 / 0 | lisible, enjoué, bascule en anglais (« Today is a fantastic day… ») |
  | ×0,05 | non | 22 / 27 | lisible, en français, plus positif que la base |

- **Verdict pour CAP-1 : suffisant.** Le cœur du projet ne passera que par le binding : il lui faut relire le vecteur en flottants (fait) et un tampon que `llama_set_adapter_cvec` applique comme la CLI (égalités à ×1 et ×0,3 sur texte dégénéré, où le vecteur domine, et 90 caractères lisibles communs à ×0,1, la preuve la plus forte). L'écart restant tient au build, pas au vecteur ; il n'affecte pas un démonstrateur qui compare des doses sur un seul chemin.
- Conséquence pour la story 1.4 : le vecteur `--method mean` n'est pas normalisé ; sur ce modèle, la plage lisible observée va d'environ ×0,05 à ×0,1, le texte dégénère dès ×0,15, et la langue bascule vers l'anglais vers ×0,1. À balayer finement sous ×0,15.

## Nommage des tenseurs (constaté et lu dans les sources b12118)

Fichier de `export_gguf` (`tools/cvector-generator`) :

- architecture GGUF `controlvector`, métadonnées `controlvector.model_hint` (ici `qwen35`) et `controlvector.layer_count` (entier ; ici 23).
- un tenseur par couche, nommé **`direction.N`**, `N` de 1 à `layer_count`, `float32` (`F32`), 1D, forme `[n_embd]`.
- **Couches absentes** : la couche 0 n'existe jamais (le chargeur la refuse : « invalid (zero) direction tensor layer index »). Qwen3.5-0.8B a 24 couches : le fichier généré contient `direction.1` à `direction.23`, la couche 24 est absente (`layer_count` = 23, soit n_couches − 1). Un fichier peut aussi omettre des couches intermédiaires : elles valent zéro.
- Le chargeur de la CLI refuse : nom hors `direction.N`, `N` = 0 ou > nombre de couches, type non F32, forme non 1D, dimensions inégales. `read_cvector` refuse les mêmes cas.
- Tampon pour `llama_set_adapter_cvec` : `n_couches × n_embd` flottants, **couche 1 en tête** (offset `n_embd × (il − 1)`), couches absentes à zéro ; appel `(ctx, données, longueur, n_embd, 1, n_couches)`. `CVector.flat()` le construit.
- Particularité Windows : `--control-vector-scaled FICHIER:ÉCHELLE` coupe au premier `:`, donc échoue avec un chemin `C:/…`. Le script passe le nom du vecteur en relatif, fixe le dossier de travail sur son dossier et passe le modèle en absolu.

## Versions et build

| Élément | Version |
|---|---|
| Générateur | `llama-cvector-generator` compilé depuis les sources llama.cpp b12118, build CPU (zig), `--method mean` |
| CLI de comparaison | `llama-completion` des binaires officiels b12118 CUDA 12.4, lancé avec `-ngl 0` |
| Binding | `llama-cpp-python` 0.3.36, roue précompilée CPU de l'index abetlen (`whl/cpu`, extra `llama`) ; expose `llama_set_adapter_cvec` |
| Bibliothèque `gguf` | résolue par `uv add gguf` (voir `uv.lock`) |
| Modèle | Qwen3.5-0.8B Q4_K_M (unsloth), `models/gguf/`, non versionné |

L'index abetlen ne publie pas de roue précompilée pour macOS x86_64 ni Windows ARM : l'extra `llama` y compile depuis les sources (compilateur C++ requis).

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
uv run python spikes/read_cvector_gguf.py equiv models/gguf/Qwen3.5-0.8B-Q4_K_M.gguf data/moods/pic13-joie.gguf --scale 1.0   # puis 0.3, 0.1, 0.05
```

Registre : ces vecteurs servent à faire exprimer une humeur au modèle (ton du texte), jamais à lui prêter un ressenti.

## Revue

### Passe 1 (2026-11-04, SHA revu 91bb011)

- Important, medium, second avis : « Équivalence : OK » surévalué (égalité seulement à ×1 sur sortie dégénérée, samplers et n_ctx non épinglés). Corrigé : samplers et `n_ctx` épinglés des deux côtés, marqueur `[end of text]` de la CLI retiré avant comparaison, comparaison de chaque chemin contre sa base, pic rejoué à ×1, ×0,3, ×0,15, ×0,1, ×0,05 ; note réécrite (prouvé / non prouvé, verdict CAP-1).
- Important, medium, second avis : `-m` relatif avec `cwd` au dossier du vecteur, la commande documentée échouait. Corrigé : modèle en absolu (vérifié avec la commande de la note).
- Important, medium, second avis : bloc `[[tool.uv.index]]` différent de celui de `exp/1-2-pic-cvec` (PR #4). Corrigé : `explicit = true` ajouté, `uv.lock` régénéré ; le rebase après #4 reste à faire par l'orchestrateur.
- Nit, low, second avis : cas réel `layer_count = n_layer − 1` non testé, `N > layer_count` non refusé. Corrigé : deux tests, refus dans `read_cvector`.
- Nit, low, second avis : pas de roue macOS x86_64 / Windows ARM. Corrigé : une ligne dans Versions.
- Nit, low, second avis : affirmations de la note sans trace. Corrigé : extrait de `show` ajouté.

### Passe 2 (2026-11-04, SHA revu 92a4a24)

- Nit, low, reviewer : verdict CAP-1 appuyé sur les seules égalités à ×1 et ×0,3, sorties dégénérées où le vecteur domine. Corrigé : la note cite aussi les 90 caractères lisibles communs à ×0,1.
- Correctifs de la passe 1 vérifiés : samplers et `n_ctx` épinglés des deux côtés, modèle en absolu, `explicit = true`, refus `N > layer_count` testé. La note distingue bien prouvé (lecture, mapping, tampon) et non prouvé (égalité entre builds).
- Résolution `pyproject.toml` / `uv.lock` avec main saine : `gguf` en dépendance de base, extra `llama` et index de main conservés, `uv lock --check` OK.
