# Brief — Démonstrateur « humeur » de SLM par steering d'activations

> Document d'entrée pour un nouveau projet. Il sert d'abord au cadrage du besoin (skills BMAD), puis au développement.
> Auteur : Anaël Yahi — rédigé le 4 novembre 2026.

---

## 1. Point de départ

Je me demande s'il est possible de **modifier l'état interne d'un LLM pour produire un équivalent de « bonheur » dans son comportement**. L'image est celle d'une molécule (ocytocine, sérotonine) qui agit sur un cerveau humain.

Deux métaphores à confronter :

| Métaphore | Équivalent technique | Nature de l'effet |
|---|---|---|
| **Chirurgie cérébrale** | Édition directe des poids (ROME/MEMIT, abliteration, fine-tuning) | Permanent, irréversible sans sauvegarde |
| **Molécule** | Steering d'activations à l'inférence (vecteur ajouté au flux résiduel) | Transitoire, dosable, réversible ; poids intacts |

Le steering est l'analogue le plus juste de la molécule. On peut même mimer une **pharmacocinétique** : le coefficient monte, atteint un pic, puis s'élimine au fil des tokens.

## 2. Ce qui est déjà établi

- **Steering d'activations** : il fonctionne. Voir ActAdd (Turner et al.) et Representation Engineering (Zou et al.).
- **Amplification de features interprétables** : démontrée sur un modèle de production avec Golden Gate Claude (Anthropic).
- **Édition de poids ciblée** : ROME/MEMIT localisent un fait et le réécrivent par une mise à jour de rang 1. L'abliteration retire une direction (le refus) directement dans les poids.
- **Post « Astra lit et écrit les poids de Qwen »** : d'après la critique de M. Honnibal, la correction de Qwen3.5-0.8B n'a pas été faite « en lisant les poids ». Astra a écrit du Python et a exécuté le modèle sur jusqu'à 50 lots de problèmes. C'est un travail d'ingénieur ML automatisé (activations → direction → édition → mesure), pas une compréhension directe des poids. La lecture à l'œil nu n'a été démontrée que sur des réseaux de 17 000 à 172 000 paramètres.
- **« Drogues pour ChatGPT » (Pharmaicy)** : ce sont seulement des prompts de jeu de rôle, sans aucune intervention sur les paramètres. C'est le contre-exemple à montrer dans la démo.
- **Paramètres d'échantillonnage** (température, top-p) : ils ne produisent aucune « humeur », ils ne jouent que sur la variabilité.

## 3. Ce que llama.cpp permet

| Fonction | Statut |
|---|---|
| Générer un vecteur de contrôle depuis un GGUF : `llama-cvector-generator` (ACP ou `--method mean`, CPU ou GPU) | ✅ Natif |
| Appliquer un vecteur : `--control-vector`, `--control-vector-scaled FNAME:SCALE`, `--control-vector-layer-range` | ✅ Natif, cumulable |
| Réglages conseillés dans la doc : couches > 10, échelle d'environ 0,8 | ✅ Point de départ |
| Changer de vecteur ou d'échelle **à chaud** via l'API de llama-server | ❌ Pas natif : fixé au lancement, redémarrage nécessaire (issue #10685) |
| Endpoint `GET/POST /cvectors` (PR #24740) | ⚠️ Proposé, statut **à vérifier** |
| Hot-swap de LoRA via `/lora-adapters` | ✅ Natif, alternative possible (mais LoRA = entraînement) |

Conséquences :

- **Ollama est exclu** : il n'expose ni les activations ni les vecteurs de contrôle. Le workbench frugal-ai-workbench actuel ne peut donc pas porter la feature tel quel.
- **Un vecteur est propre à un modèle** (architecture, dimension cachée, couche visée). Il doit être généré sur le même GGUF, avec la même quantification, que celui qu'on sert.
- **Aucun entraînement n'est nécessaire** : on fait une **calibration** sans gradient (environ 100 paires de prompts contrastées, quelques minutes sur un SLM). Elle se fait une fois par couple modèle × humeur, et le résultat se met en cache.

## 4. Options d'architecture à explorer

| Option | Principe | Pour | Contre |
|---|---|---|---|
| **A. llama-swap + llama-server** | Une entrée de config par modèle × humeur × intensité ; le front choisit via le champ `model` (API OpenAI) | Zéro code serveur, mise en route rapide | Paliers discrets ; rechargement du modèle à chaque changement ; nombre de configs qui grossit ; RAM × N si plusieurs variantes restent chargées |
| **B. In-process (llama-cpp-python, API C bas niveau)** | Appliquer ou retirer le vecteur entre deux requêtes, sans recharger | Curseur continu, réactif ; reste sur GGUF | Plus de code ; disponibilité de l'API d'application du vecteur dans les bindings **à vérifier** |
| **C. transformers + nnsight / TransformerLens** | Hooks PyTorch sur les activations | Flexibilité maximale, visualisation de l'intervention, pharmacocinétique token par token facile | Plus lent sur CPU ; s'éloigne d'une stack d'inférence « de production » |
| **D. Édition de poids (« chirurgie »)** | Inscrire la direction « joie » dans les poids (type abliteration inversée) | Montre la différence permanent / transitoire | Irréversible ; un fichier modèle par humeur |

Piste par défaut : **A** pour valider l'effet rapidement, **B** si le curseur continu est indispensable, **C/D** en module pédagogique.

## 5. Hypothèses à tester (pour me construire des convictions)

1. **Effet perceptible** : sur un SLM de 1 à 4B, un vecteur « joie » produit-il un changement de ton net et cohérent avant que le texte ne se dégrade ? Quelle est la plage d'échelle utile par modèle ?
2. **Taille du modèle** : l'effet devient-il plus fin et plus stable en passant de 0.8B à 7-9B ?
3. **Coût sur la tâche** : quel impact sur l'exactitude (QCM simple, RAG, tool calling) ? Le steering « joie » augmente-t-il la complaisance (le modèle contredit-il moins) ?
4. **Steering vs prompt** : à ton comparable, qu'apporte le steering face à un system prompt « sois joyeux » ? Plus robuste face au jailbreak ? Plus subtil ? Plus coûteux ?
5. **Spécificité** : la direction « joie » est-elle distincte de « politesse » ou « enthousiasme » ? Les humeurs se combinent-elles (joie + calme) ?
6. **Pharmacocinétique** : une décroissance du coefficient au fil des tokens donne-t-elle un effet « montée / redescente » lisible ?
7. **Frugalité** : surcoût en latence, en énergie et en CO₂ (CodeCarbon) du steering face au prompt.

## 6. Démonstrateur envisagé (à affiner avec BMAD)

- Un front simple : choix du modèle, sélecteur d'humeur, curseur d'intensité.
- Un **mode comparaison** côte à côte : neutre / prompt « joyeux » / steering « joie ».
- Un script de calibration : paires contrastées → vecteur GGUF par modèle, rangé dans `data/moods/{model_id}/{humeur}.gguf`.
- Un petit banc d'évaluation : ton (LLM-as-a-judge ou classifieur de sentiment), exactitude sur tâche, complaisance, mesure CodeCarbon.
- Un repli : system prompt si aucun vecteur n'existe pour le modèle chargé.

## 7. Questions ouvertes pour le cadrage BMAD

- **Objectif principal** : conviction personnelle, support de formation (veille, task force IA), ou briques réutilisables dans mes autres applications SLM ?
- **Public de la démo** : moi seul, des collègues consultants, des clients ?
- **Périmètre** : steering seul, ou aussi le volet « chirurgie » (édition de poids) pour le contraste ?
- **Modèles cibles** et contraintes matérielles (portable, CPU/GPU, RAM).
- **Humeurs** : « joie » seule pour le MVP, ou plusieurs dès le départ ?
- **Critères de succès** mesurables pour chacune des hypothèses de la section 5.
- **Intégration future** dans frugal-ai-workbench (migration Ollama → llama-server) : oui, non, plus tard ?

## 8. Points de vigilance

- **Comportement ≠ ressenti** : le démonstrateur montre une *expression* d'humeur. Il ne dit rien d'un éventuel état vécu. Le formuler ainsi dans toute communication.
- Un steering trop fort dégrade la cohérence et l'exactitude.
- Le risque de complaisance accrue est à mesurer, pas seulement à supposer.
- Plusieurs points restent à vérifier dans la version courante de llama.cpp : statut de la PR #24740, API de vecteur de contrôle dans llama-cpp-python.

## 9. Sources

- ActAdd — Steering Language Models With Activation Engineering : https://arxiv.org/abs/2308.10248
- Representation Engineering : https://arxiv.org/abs/2310.01405
- ROME — Locating and Editing Factual Associations in GPT : https://arxiv.org/abs/2202.05262
- MEMIT — Mass-Editing Memory in a Transformer : https://arxiv.org/abs/2210.07229
- Refusal is mediated by a single direction (abliteration) : https://arxiv.org/abs/2406.11717
- Golden Gate Claude (Anthropic) : https://www.anthropic.com/news/golden-gate-claude
- Post original « Weight-space programming » (Astra / Qwen) : https://rikiyat.github.io/blog/weight-space-programming/
- Critique de M. Honnibal : https://honnibal.dev/blog/astra-read-write-misleading
- Pharmaicy (drogues simulées par prompt) : https://aitopics.org/tag/ruddwall et https://www.milleworld.com/this-new-platform-lets-you-drug-your-ai-chatbot/
- README cvector-generator (llama.cpp) : https://sources.debian.org/src/llama.cpp/7965%2Bdfsg-1/tools/cvector-generator/README.md
- Options de llama-server (man page) : https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html
- Issue llama.cpp #10685 (hot-swap des vecteurs de contrôle) : https://github.com/ggml-org/llama.cpp/issues/10685
- llama-swap : https://github.com/mostlygeek/llama-swap
- Workbench existant : https://github.com/Aliquanto3/frugal-ai-workbench
