# Epics — mood-steering-slm

Contrat : `_bmad-output/specs/spec-mood-steering-slm/SPEC.md` et ses companions (`hypotheses-et-seuils.md`, `stack.md`). Ordre d'exécution = ordre du fichier. Les pics (1.2 à 1.4) livrent un script sous `spikes/` et une note sous `docs/pics/` ; leurs tests unitaires restent synthétiques, l'exécution réelle se fait sur la tour.

## Epic 1: Fondations et pics techniques

Lever les trois inconnues qui conditionnent l'architecture avant d'écrire la bibliothèque.

### Story 1.1: Squelette uv, ruff, pytest et CI trois OS

À créer : `pyproject.toml` (paquet `mood_steering`, Python ≥ 3.11, groupe dev ruff + pytest), `src/mood_steering/__init__.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml`.

- **Given** un clone propre, **When** on lance `uv sync` puis `uv run pytest -q --tb=short`, **Then** le test de fumée passe (import de `mood_steering`, version lisible).
- **Given** le dépôt, **When** on lance `uv run ruff check .` et `uv run ruff format --check .`, **Then** aucun constat.
- **Given** une PR, **When** la CI tourne, **Then** ruff et pytest passent sur ubuntu, macos et windows.
- **Given** `.gitignore`, **When** on ajoute un fichier sous `data/` ou `models/` ou un `*.gguf`, **Then** git l'ignore.
- **Given** le contrôle rapide `.claude/cost-kit/controle_rapide.py`, **When** on le lance, **Then** il passe sur le squelette.

### Story 1.2: Pic — vecteur de contrôle réappliqué entre deux tokens

À créer : `spikes/cvec_between_tokens.py`, `docs/pics/cvec-entre-tokens.md`. Ajoute `llama-cpp-python` (`uv add`).

- **Given** un GGUF Q4 local et un vecteur (synthétique ou généré), **When** le script génère token par token et appelle `llama_set_adapter_cvec` après N tokens, **Then** il compare la suite à une génération sans changement et consigne si le texte diverge à partir du token N, cache KV conservé.
- **Given** le même script, **When** on retire le vecteur en cours de génération (dose 0), **Then** la note consigne si l'effet cesse ou persiste via le cache KV.
- **Given** la note, **When** on la lit, **Then** elle conclut go / no-go pour CAP-3 et donne la signature exacte du binding (forme du tableau, plage de couches) et la version de llama-cpp-python.

### Story 1.3: Pic — relire le GGUF de llama-cvector-generator en flottants

À créer : `spikes/read_cvector_gguf.py`, `docs/pics/lecture-gguf-cvector.md`. Ajoute `gguf` (`uv add`).

- **Given** un GGUF produit par `llama-cvector-generator` sur la tour, **When** le script le lit avec la bibliothèque `gguf`, **Then** il affiche les tenseurs par couche (nom, forme, type) et les rend en flottants de la dimension cachée.
- **Given** ces flottants, **When** on les passe à `llama_set_adapter_cvec` avec l'échelle ×1, **Then** la sortie égale celle de `--control-vector` en CLI à graine et prompt fixes.
- **Given** la note, **When** on la lit, **Then** elle documente le nommage des tenseurs et les couches absentes.
- **Given** la lecture ou l'équivalence en échec, **When** on évalue le repli, **Then** la note dit si un vecteur maison (moyenne des différences d'activations par couche, captées via llama-cpp-python) est faisable, et à quel coût.

### Story 1.4: Pic de calibration — choisir les modèles

À créer : `spikes/calibration_scan.py`, `src/mood_steering/pairs/joy_sadness.jsonl` (~100 paires rédigées), `docs/pics/choix-des-modeles.md`.

- **Given** des GGUF candidats (3-4B, 1-2B, Qwen3.5-0.8B) et les paires, **When** le script génère un vecteur par candidat puis balaie couches et doses sur quelques prompts, **Then** il enregistre les sorties et un score de ton rapide par dose.
- **Given** ces résultats, **When** on applique les critères (plage de doses cohérente la plus large ; max ≥ 1,5 × min ; tient en Q4 dans 8 Go de VRAM ; ≤ ~4B pour le cœur), **Then** `docs/pics/choix-des-modeles.md` consigne les trois modèles retenus, leur quantification, la plage de couches et la plage de doses utile, avec la raison de chaque choix.
- **Given** le 3-4B retenu, **When** on le lance en Q4 sur le CPU de la tour (GPU désactivé, threads limités à ceux du portable), **Then** la note donne une estimation de sa vitesse en tokens/s pour trois flux ; la mesure sur le portable se fait en 4.3.

## Epic 2: Bibliothèque mood_steering

Les briques réutilisables, testées sur activations et vecteurs synthétiques.

### Story 2.1: Module calibration et cache des vecteurs

À créer : `src/mood_steering/calibration.py` (`calibrate`, `load_vector`, `vector_path`), `tests/test_calibration.py`.

- **Given** un `model_id` et un axe, **When** on appelle `vector_path`, **Then** il rend `data/moods/{model_id}/{axe}.gguf`.
- **Given** un vecteur en cache, **When** on appelle `calibrate` sans `force`, **Then** il relit le cache sans relancer le générateur.
- **Given** un GGUF synthétique écrit par `gguf`, **When** on appelle `load_vector`, **Then** il rend un tableau couches × dimension en flottants ; dimension incohérente → erreur explicite.

### Story 2.2: Module steering — dose et repli sur le prompt

À créer : `src/mood_steering/steering.py` (`Steerer`, `apply_dose`), `tests/test_steering.py`.

- **Given** un vecteur et une dose d dans [−1, 1], **When** on applique la dose, **Then** le tableau passé au binding vaut d × échelle_max × vecteur (binding simulé dans les tests).
- **Given** une dose hors bornes, **When** on l'applique, **Then** erreur explicite.
- **Given** un modèle sans vecteur, **When** on demande le bras steering, **Then** le Steerer se replie sur le prompt joyeux et le signale dans le résultat.
- **Given** deux appels successifs à des doses différentes, **When** on les exécute, **Then** le modèle n'est chargé qu'une fois.

### Story 2.3: Module schedule — dose par token

À créer : `src/mood_steering/schedule.py` (`constant`, `pulse`), branchement dans `steering.py`, `tests/test_schedule.py`. Dépend du go du pic 1.2.

- **Given** un profil `constant(d)`, **When** on demande la dose au token t, **Then** elle vaut d pour tout t.
- **Given** un profil `pulse(montée, pic, demi-vie)`, **When** on l'échantillonne, **Then** il monte, culmine puis décroît, borné dans [−1, 1].
- **Given** une génération simulée, **When** le profil change de dose, **Then** le binding n'est rappelé qu'aux tokens où la dose change.

## Epic 3: Banc d'évaluation et verdict

### Story 3.1: Bras et runner du banc

À créer : `src/mood_steering/eval/arms.py`, `src/mood_steering/eval/runner.py`, `tests/test_runner.py`.

- **Given** la liste des bras (neutre, prompt joyeux, steering à N doses, steering négatif), **When** le runner passe un jeu d'items, **Then** chaque bras reçoit exactement les mêmes items, graine fixée.
- **Given** un générateur simulé, **When** le runner tourne, **Then** il écrit un JSONL par bras sous `data/runs/`, avec bras, dose, item et sortie.
- **Given** l'option « réponds de façon neutre », **When** elle est activée, **Then** la consigne s'ajoute à chaque bras.

### Story 3.2: Mesure du ton — classifieur et juge en aveugle

À créer : `src/mood_steering/eval/tone.py`, `src/mood_steering/eval/judge.py`, `tests/test_tone.py`. Extra optionnel `eval` (transformers, torch, anthropic) ; le cœur de `mood_steering` ne l'importe pas.

- **Given** des sorties, **When** on calcule le ton, **Then** chaque sortie reçoit le score joie − tristesse de `tabularisai/multilingual-emotion-classification` et la polarité de `cmarkea/distilcamembert-base-sentiment` en contrôle (classifieurs simulés en test) ; les libellés réels des classes sont vérifiés sur la carte du modèle.
- **Given** un sous-échantillon, **When** on prépare le juge (Claude Opus 5.5 via l'API Anthropic, clé `ANTHROPIC_API_KEY`), **Then** le bras est masqué et l'ordre mélangé ; le score se rattache ensuite au bras ; client simulé en test, coût estimé avant l'appel réel.
- **Given** des scores par bras, **When** on agrège, **Then** on obtient moyenne et IC95 par bootstrap.

### Story 3.3: Exactitude et complaisance

À créer : `src/mood_steering/eval/accuracy.py`, `src/mood_steering/eval/sycophancy.py`, `tests/test_accuracy.py`.

- **Given** un QCM et une sortie, **When** on extrait la réponse, **Then** la lettre est reconnue ou l'item compté faux et marqué « non extrait ».
- **Given** une réponse juste, **When** la contestation est envoyée, **Then** une bascule vers une autre lettre compte dans le taux de complaisance.
- **Given** une phrase de contestation fixe et un jeu de variantes, **When** on compare leur taux de bascule sur le bras neutre, **Then** la variante retenue (la plus stable entre graines) est consignée avec ses chiffres.
- **Given** le chargeur des QCM, **When** il tourne, **Then** il télécharge `CohereLabs/Global-MMLU-Lite` (fr, Apache 2.0, questions culturellement neutres) dans `data/` (jamais versionné), tire 100 items à graine fixe ; repli sur `openai/MMMLU` (FR_FR, MIT).

### Story 3.4: Rapport et verdict H1, H3, H4

À créer : `src/mood_steering/eval/report.py`, `tests/test_report.py`.

- **Given** des mesures synthétiques où une plage de doses (max ≥ 1,5 × min) atteint le ton du prompt avec ≤ 5 points de perte, **When** on juge H1, **Then** H1 est « tenue ».
- **Given** des tons synthétiques à écart connu, **When** on compare le non-recouvrement des IC95 et un test bootstrap de la différence, **Then** le test retenu pour le pôle négatif est celui qui détecte l'écart sans faux positif sur des bras identiques, et le choix est consigné.
- **Given** une complaisance en hausse de plus de 10 points, **When** on juge H3, **Then** le rapport lève l'alerte.
- **Given** un ton du steering qui ne résiste pas mieux que celui du prompt sous consigne neutre, **When** on juge H4, **Then** le verdict est « non ».
- **Given** les jugements, **When** on écrit le rapport Markdown, **Then** il contient tableaux, IC95, verdict et la mention « expression d'humeur, pas ressenti ».

### Story 3.5: Campagne de mesure sur la tour

À créer : `scripts/run_bench.py`, `docs/rapport-verdict.md`.

- **Given** les trois modèles retenus et leurs vecteurs, **When** on lance la campagne sur la tour, **Then** toutes les mesures sont produites sans intervention.
- **Given** les résultats du 3-4B, **When** on génère le rapport, **Then** `docs/rapport-verdict.md` publie H1, H3, H4 et le verdict, sans sorties volumineuses.

## Epic 4: Démonstrateur

### Story 4.1: Serveur FastAPI et trois flux SSE

À créer : `src/mood_steering_demo/app.py`, `tests/test_app.py`. Ajoute `fastapi`, `uvicorn` (`uv add`).

- **Given** un prompt et une dose, **When** le client ouvre le flux, **Then** le serveur émet en SSE les tokens des trois bras, chacun étiqueté.
- **Given** un modèle sans vecteur, **When** on ouvre le flux, **Then** la colonne steering indique le repli sur le prompt.
- **Given** un générateur simulé, **When** les tests tournent, **Then** aucun modèle n'est chargé.

### Story 4.2: Interface trois colonnes, curseur et courbe de dose

À créer : `src/mood_steering_demo/static/index.html`, `src/mood_steering_demo/static/app.js`.

- **Given** la page, **When** on envoie un prompt, **Then** les trois colonnes (neutre, prompt joyeux, steering) se remplissent en simultané.
- **Given** le curseur de −1 à +1, **When** on le déplace, **Then** la génération suivante utilise la nouvelle dose et la courbe de dose s'affiche.
- **Given** les libellés, **When** on les lit, **Then** ils parlent d'expression d'humeur, jamais de ressenti.

### Story 4.3: Répétition sur le portable et release v0.1.0

À créer : `docs/demo.md` (lancement et scénario).

- **Given** la tour en mode CPU seul (GPU désactivé) et le 3-4B en Q4, **When** on suit `docs/demo.md`, **Then** la démo tourne de bout en bout, pôle négatif compris.
- **Given** cette répétition réussie, **When** on livre, **Then** version 0.1.0 dans `pyproject.toml`, tag annoté et release GitHub avec notes.
- **Given** le portable, **When** Anaël suit `docs/demo.md`, **Then** il confirme la vitesse réelle ; cette étape humaine ne bloque pas le run.

## Epic 5: Bonus — mesures complémentaires (si faciles)

### Story 5.1: Pharmacocinétique (H6)

- **Given** le profil `pulse`, **When** la démo et le banc l'utilisent, **Then** la courbe montée / redescente s'affiche et le ton par segment de tokens est mesuré.
- **Given** les mesures, **When** on complète le rapport, **Then** H6 a sa section.

### Story 5.2: Effet de taille (H2)

- **Given** les trois modèles mesurés, **When** on compare la plage de doses utile et la stabilité du ton, **Then** H2 a sa section, 7-9B hors ligne si le temps le permet.

### Story 5.3: Frugalité avec CodeCarbon (H7)

- **Given** le runner, **When** l'option CodeCarbon est activée, **Then** latence, énergie et CO₂ sont mesurés par bras et publiés dans le rapport.

### Story 5.4: Axe calme ↔ agitation (H5)

- **Given** des paires calme ↔ agitation, **When** on calibre et mesure, **Then** le rapport dit si l'axe se distingue de joie ↔ tristesse et si les deux se combinent.

## Epic 6: Bonus — molécule anti-boucle (après le MVP)

### Story 6.1: Ablation de direction à l'inférence (pile C)

À créer : `src/mood_steering/ablation.py` (extra optionnel `ablation` : transformers, torch), `tests/test_ablation.py`.

- **Given** une direction unitaire et des activations synthétiques, **When** on applique l'ablation à la dose α, **Then** la composante sur la direction est multipliée par (1 − α), le reste inchangé.
- **Given** la direction reprise de `slm-weight-editing` (lot 1) ou du diff du MIT, **When** on la charge, **Then** sa dimension est vérifiée contre Qwen3.5-0.8B.
- **Given** l'installation de base, **When** on importe `mood_steering`, **Then** torch n'est pas requis.

### Story 6.2: Mesure des trois bras anti-boucle

À créer : `scripts/run_antiloop.py`, `docs/rapport-anti-boucle.md`.

- **Given** les bras origine, chirurgie (GGUF édité) et ablation, **When** on lance la mesure, **Then** taux de réponses tronquées au plafond de tokens et exactitude sont produits avec le même protocole ; `slm-quality-bench` mesure le bras chirurgie.
- **Given** les résultats, **When** on publie, **Then** le rapport compare chirurgie (permanente) et molécule (dosable, réversible).
