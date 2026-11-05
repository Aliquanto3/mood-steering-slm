# Run autonome — plan et état

Fichier d'état de l'orchestrateur, à relire en premier après chaque `/clear`. Lu par `/agent-cost-kit:avancement-run`. Machine du run : la tour (RTX 3050, 8 Go), qui exécute aussi les pics et la campagne.

## Périmètre

| Périmètre | Epics | Note |
|---|---|---|
| Dans le run | Epics 1 à 6 | MVP (1 à 4), puis bonus (5 et 6) ajoutés par Anaël le 2026-11-04 avant la nuit ; les bonus ne passent jamais devant une story du MVP |

Au plus 2 stories en développement à la fois, chacune dans son worktree.
Plafond global : 5 agents actifs à la fois sous Windows (développement, revue et rebase confondus).

## Règles

- Types : `dev` (dev-story → second-avis → reviewer → intégration), `mixte` (code par l'agent, puis « attente humaine »), `humaine` (jamais confiée à un agent). ★ = story pivot, dev-story sur Opus.
- Petites stories bien décrites dans `epics.md` : dev-story directement sur l'entrée d'epics, sans fichier de story BMAD (règle expérimentale d'aiguillage, effet noté au journal).
- Deux stories du même fichier partagé ne vont jamais dans la même vague, sauf `pyproject.toml` / `uv.lock` (ajout de dépendance) : la seconde PR à fusionner rebase et relance `uv lock`.
- Fichiers d'état du run (ce fichier, `sprint-status.yaml`) : jamais dans une PR de story. PR `chore/etat-run` à part.
- Suite complète : pas en local pendant un run parallèle, la CI de branche la joue et sert de porte de fusion. Attente : `gh run watch <id> --interval 60`.
- Fusion par merge commit (`gh pr merge --merge`), jamais de push sur `main`. Journaliser `fusion` seulement après avoir vérifié `MERGED`, puis `git log -1` sur `main`.
- Mise à jour d'une branche de story sur `main` : `git merge origin/main` puis push simple ; jamais de rebase suivi d'un push forcé (refusé par le classifieur le 2026-11-05).
- `git worktree prune` avant chaque lancement. Les worktrees ne sont supprimés qu'en fin de run.
- Heure du journal : toujours prise par `date` au moment d'écrire, jamais devinée. `uv run python` sous Windows.
- Ressources humaines : `ANTHROPIC_API_KEY` absente au lancement (juge de 3.2 simulé en test ; appel réel en 3.5) ; test sur le portable (4.3), non bloquant. Prévenir Anaël par notification et avancer sur le reste.

## Sessions autorisées à modifier le périmètre

- (aucune)

## Contrats partagés en cours

- Paquet `mood_steering` sous `src/` (layout src), Python ≥ 3.11, version lue par `mood_steering.__version__`.
- `llama-cpp-python` en extra optionnel `llama` (`uv add --optional llama llama-cpp-python`) : la CI ne l'installe pas, les tests simulent le binding. `gguf` en dépendance de base (pur Python).
- Extra optionnel `eval` (transformers, torch, anthropic) pour 3.2 ; le cœur ne l'importe pas.
- Clé du juge : `ANTHROPIC_API_KEY` lue dans l'environnement, avec repli sur le `.env` du dépôt principal `C:/Users/Utilisateur/Documents/GitHub/mood-steering-slm/.env` (ignoré par git, absent des worktrees ; la variable globale a été retirée par Anaël), chargé par `python-dotenv` dans l'extra `eval`. Jamais affichée ni journalisée.
- Binaires llama.cpp : build officiel `b12118` CUDA 12.4 dans `models/bin/llama-b12118-cuda12.4/` (sans `llama-cvector-generator`, absent des builds publiés) ; `llama-cvector-generator` compilé sur CPU depuis les sources `b12118` dans `models/src/llama.cpp/build-zig/bin/` (zig via `uv tool`, wrappers dans `models/bin/zigwrap/`, manifeste Windows sauté par une option locale `LLAMA_SKIP_WIN_MANIFEST` ajoutée à `cmake/win-app.cmake`).
- Extra optionnel `ablation` (transformers, torch) pour 6.1 ; direction anti-boucle reprise du dépôt voisin `../slm-weight-editing`.
- Modèles téléchargés dans `models/`, vecteurs dans `data/moods/{model_id}/{axe}.gguf`, sorties du banc dans `data/runs/` : jamais versionnés.

## Sprint 1 — Fondations et pics (Epic 1)

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 1.1 ★ | — | `pyproject.toml`, `src/mood_steering/__init__.py`, `.github/workflows/ci.yml`, `.gitignore` |
| V1 | 1.2 (mixte) · 1.3 (mixte) | 1.2 ← 1.1 ; 1.3 ← 1.1 | 1.2 `spikes/cvec_between_tokens.py`, `docs/pics/cvec-entre-tokens.md` ; 1.3 `spikes/read_cvector_gguf.py`, `docs/pics/lecture-gguf-cvector.md` ; les deux `pyproject.toml` |
| V2 | 1.4 ★ (mixte) · 6.1 | 1.4 ← 1.2, 1.3 ; 6.1 ← 1.1 | 1.4 `spikes/calibration_scan.py`, `src/mood_steering/pairs/joy_sadness.jsonl`, `docs/pics/choix-des-modeles.md` ; 6.1 `ablation.py`, `pyproject.toml` |

6.1 (bonus, tests synthétiques seulement) prend la place libre de V2 pendant que 1.4 occupe le GPU.

`mixte` ici : l'agent code puis exécute lui-même le pic sur la tour ; pas d'attente humaine prévue.

## Sprint 2 — Bibliothèque et banc (Epics 2 à 3)

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 2.1 · 2.2 | 2.1 ← 1.3 ; 2.2 ← 1.2 | 2.1 `calibration.py` ; 2.2 `steering.py` |
| V1 | 2.3 · 3.1 | 2.3 ← 1.2, 2.2 ; 3.1 ← 2.2 | 2.3 `schedule.py`, `steering.py` ; 3.1 `eval/arms.py`, `eval/runner.py` |
| V2 | 3.2 · 3.3 | 3.2 ← 3.1 ; 3.3 ← 3.1 | 3.2 `eval/tone.py`, `eval/judge.py`, `pyproject.toml` ; 3.3 `eval/accuracy.py`, `eval/sycophancy.py` |
| V3 | 3.4 · 4.1 | 3.4 ← 3.2, 3.3 ; 4.1 ← 2.2 | 3.4 `eval/report.py` ; 4.1 `mood_steering_demo/app.py`, `pyproject.toml` |
| V4 | 3.5 ★ (mixte) · 4.2 | 3.5 ← 3.4, 1.4 ; 4.2 ← 4.1 | 3.5 `scripts/run_bench.py`, `docs/rapport-verdict.md` ; 4.2 `static/index.html`, `static/app.js` |

## Sprint 3 — Démonstrateur (Epic 4)

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 4.3 (mixte) | 4.3 ← 3.5, 4.2 | `docs/demo.md`, `pyproject.toml` (v0.1.0) |

## Sprint 4 — Bonus (Epics 5 à 6)

Les stories 5.x n'ont pas de noms de fichiers à créer dans `epics.md` : cadrage court (fichier de story) avant le dev-story. Une hypothèse non atteinte est marquée « non traitée » dans le rapport (CAP-8).

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 5.1 · 5.3 | 5.1 ← 2.3, 3.5, 4.2 ; 5.3 ← 3.5 | 5.1 démo (`app.js`), `eval/`, rapport ; 5.3 `eval/runner.py`, `pyproject.toml` |
| V1 | 5.2 · 6.2 | 5.2 ← 3.5 ; 6.2 ← 6.1, 3.3 | 5.2 rapport ; 6.2 `scripts/run_antiloop.py`, `docs/rapport-anti-boucle.md` |
| V2 | 5.4 (mixte) | 5.4 ← 2.1, 3.5 | paires calme ↔ agitation, rapport |

Le rapport `docs/rapport-verdict.md` est touché par 5.1, 5.2 et 5.4 : chaque story ajoute sa propre section ; la seconde à fusionner rebase.

« Après » ne lit que `x ← y` ou `Epics N à M`.

## Journal

Une ligne par étape : `- AAAA-MM-JJ HH:MM · story · étape · résultat · commit`. Une story est terminée quand sa PR est fusionnée.

- 2026-11-04 23:24 · — · décision · sprint planning : gate PASS, 6 epics, 21 stories, run Epics 1 à 4, 2 stories en parallèle au plus · —
- 2026-11-04 23:26 · 1.1 · dev · lancé (dev-story Opus, worktree, sans cadrage BMAD) · —
- 2026-11-04 23:27 · 1.1 · dev · fini, 2 tests verts, ruff et contrôle rapide OK · cc89529
- 2026-11-04 23:30 · — · décision · Anaël ajoute les Epics 5 et 6 au run (bonus, après le MVP ; 6.1 en V2 du sprint 1) · —
- 2026-11-04 23:30 · 1.1 · second-avis · 0 medium/high, 5 low · —
- 2026-11-04 23:30 · 1.1 · review · conforme, 3 low corrigés (.blocages, timeout/concurrency CI, hatchling>=1.27) · 77ae95c
- 2026-11-04 23:32 · 1.1 · PR · #3 ouverte · 77ae95c
- 2026-11-04 23:32 · 1.1 · CI · verte sur les trois OS · 77ae95c
- 2026-11-04 23:32 · 1.1 · fusion · #3 fusionnée, MERGED vérifié · 401d5e4
- 2026-11-04 23:32 · 1.2 · dev · lancé (dev-story Sonnet, worktree) · —
- 2026-11-04 23:32 · 1.3 · dev · lancé (dev-story Sonnet, worktree) · —
- 2026-11-04 23:38 · 1.2 · dev · fini, GO pour CAP-3 (divergence au token N+1, rémanence après retrait), 17 tests verts · 9f5a4bc
- 2026-11-04 23:38 · — · décision · clé du juge dans .env (ignoré), variable globale retirée · —
- 2026-11-04 23:40 · 1.2 · second-avis · 2 medium (rémanence non séparée, GO à une seule dose dégénérée), 5 low · —
- 2026-11-04 23:43 · — · décision · llama-cvector-generator compilé sur CPU (zig, b12118, sans manifeste) · —
- 2026-11-04 23:48 · 1.3 · dev · fini, lecture OK (direction.1..23, F32), équivalence ×1 au caractère mais sortie dégénérée, 10 tests verts · 91bb011
- 2026-11-04 23:49 · 1.2 · review · passe 1 : 2 medium et 5 low corrigés, GO à norme 0,6, retirer suffit · 5afb5b5
- 2026-11-04 23:50 · 1.2 · PR · #4 ouverte · 5afb5b5
- 2026-11-04 23:50 · 1.2 · review · passe 2 : 1 medium de doc corrigé (deux sources de rémanence), vérifié par l'orchestrateur, pas de passe 3 · c65eeb9
- 2026-11-04 23:50 · 1.2 · CI · verte sur les trois OS · c65eeb9
- 2026-11-04 23:50 · 1.2 · fusion · #4 fusionnée, MERGED vérifié · 9054c94
- 2026-11-04 23:50 · 1.3 · second-avis · 3 medium (équivalence surévaluée, chemin modèle relatif, conflit lock avec 1.2), 3 low · —
- 2026-11-04 23:50 · 1.3 · review · lancée (passe 1) · —
- 2026-11-04 23:51 · 2.2 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD ; avancée devant 1.4 et 6.1 car le MVP passe d'abord) · —
- 2026-11-04 23:53 · 2.2 · dev · fini, 41 tests verts, numpy en base, backend réel non exécuté · 48ea651
- 2026-11-04 23:55 · 2.2 · second-avis · 1 high (préfixe KV réutilisé entre doses : dose 0 ≠ neutre), 3 medium, 1 low · —
- 2026-11-04 23:57 · 1.3 · review · passe 1 : 3 medium et 3 low corrigés, CAP-1 suffisant (binding seul), doses mean utiles ×0,05 à ×0,1 ; rebase sur main par l'orchestrateur, passe 2 lancée · 92a4a24
- 2026-11-05 00:01 · 1.3 · review · passe 2 : aucun Important, 1 low de doc corrigé · 1f6de41
- 2026-11-05 00:01 · 1.3 · PR · #5 ouverte · 1f6de41
- 2026-11-05 00:01 · 1.3 · CI · verte sur les trois OS · 1f6de41
- 2026-11-05 00:01 · 1.3 · fusion · #5 fusionnée, MERGED vérifié · 15b1293
- 2026-11-05 00:01 · 2.2 · review · passe 1 : 1 high (reset du contexte KV), 3 medium, 1 low corrigés ; défaut non reproduit sur Qwen3.5 hybride mais réel en attention pure · a2becae
- 2026-11-05 00:01 · 2.2 · décision · main fusionné dans la branche (push forcé refusé), conflit du lock résolu, 61 tests verts · 31f2f77
- 2026-11-05 00:01 · 2.2 · review · passe 2 lancée · —
- 2026-11-05 00:01 · 1.4 · dev · lancé (dev-story Opus, pivot, worktree) · —
- 2026-11-05 00:04 · 2.2 · review · passe 2 : 0 constat, tests rouges sans les correctifs, reset() suffisant · 31f2f77
- 2026-11-05 00:04 · 2.2 · PR · #6 ouverte · 31f2f77
- 2026-11-05 00:04 · 2.2 · CI · verte sur les trois OS · 31f2f77
- 2026-11-05 00:04 · 2.2 · fusion · #6 fusionnée, MERGED vérifié · 964f998
- 2026-11-05 00:04 · 2.1 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD) · —
- 2026-11-05 00:06 · 2.1 · dev · fini, 22 tests ajoutés (83 au total), appel réel du générateur non exécuté · f346d39
- 2026-11-05 00:06 · 2.1 · second-avis · lancé · —
- 2026-11-05 00:08 · 2.1 · second-avis · 3 medium (paires absentes avant 1.4, retours à la ligne, cache corrompu), 4 low · —
- 2026-11-05 00:08 · 2.1 · review · passe 1 lancée · —
- 2026-11-05 00:14 · 2.1 · review · passe 1 : 2 medium et 4 low corrigés, 1 low écarté ; appel réel du générateur validé (~26 s) · 2a01d77
- 2026-11-05 00:14 · 2.1 · PR · #7 ouverte · 2a01d77
- 2026-11-05 00:14 · 2.1 · review · passe 2 lancée · —
- 2026-11-05 00:17 · 2.1 · review · passe 2 : 0 constat, échappement conforme au générateur, tests rouges sans correctifs · 2a01d77
- 2026-11-05 00:17 · 2.1 · CI · verte sur les trois OS · 2a01d77
- 2026-11-05 00:17 · 2.1 · fusion · #7 fusionnée, MERGED vérifié · 14f2783
- 2026-11-05 00:17 · 3.1 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD ; passe devant 2.3, chemin critique du banc) · —
- 2026-11-05 00:19 · 3.1 · dev · fini, 8 tests (98 au total) ; signale que sans vecteur le bras neutre reçoit le prompt joyeux · 81bae08
- 2026-11-05 00:19 · 3.1 · second-avis · lancé · —
- 2026-11-05 00:20 · 3.1 · second-avis · 2 high (bras neutre joyeux sans vecteur), 2 medium (écrasement, écriture non atomique), 2 low · —
- 2026-11-05 00:20 · 3.1 · décision · contrat du repli : seulement à dose ≠ 0 ; dose 0 jamais modifiée ; dose < 0 sans vecteur → SAD_SYSTEM_PROMPT · —
- 2026-11-05 00:20 · 3.1 · review · passe 1 lancée · —
- 2026-11-05 00:24 · 3.1 · review · passe 1 : 2 high, 2 medium, 2 low corrigés ; SAD_SYSTEM_PROMPT et ARM_PROMPT_SAD ajoutés à steering.py · 6b3a68a
- 2026-11-05 00:24 · 3.1 · PR · #8 ouverte · 6b3a68a
- 2026-11-05 00:24 · 3.1 · review · passe 2 lancée · —
- 2026-11-05 00:25 · 3.1 · review · passe 2 : aucun Important, 3 low notés (overwrite large, .part conservé, run partiel) · 6b3a68a
- 2026-11-05 00:25 · 3.1 · CI · verte sur les trois OS · 6b3a68a
- 2026-11-05 00:25 · 3.1 · fusion · #8 fusionnée, MERGED vérifié · 1078f3b
- 2026-11-05 00:25 · 3.2 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD) · —
- 2026-11-05 00:33 · 3.2 · dev · fini, 21 tests (137 au total) ; classifieurs réels essayés sur la tour (joie 0,97 / neutre 0,00 / triste −0,99) ; juge : 200 sorties, ~0,30 $ estimés, aucun appel réel · 6050f12
- 2026-11-05 00:33 · 3.2 · second-avis · lancé · —
- 2026-11-05 00:35 · 3.2 · second-avis · 0 high, 1 medium (chemin absolu du .env), 5 low ; IC95 vérifié comme vrai percentile · —
- 2026-11-05 00:35 · 3.2 · review · passe 1 lancée · —
- 2026-11-05 00:41 · 3.2 · review · passe 1 : 1 medium et 5 low corrigés ; correctif vérifié par l'orchestrateur, pas de passe 2 · 24298b3
- 2026-11-05 00:41 · 3.2 · PR · #9 ouverte · 24298b3
- 2026-11-05 00:41 · 3.2 · CI · verte sur les trois OS · 24298b3
- 2026-11-05 00:41 · 3.2 · fusion · #9 fusionnée, MERGED vérifié · bd4cbe5
- 2026-11-05 00:41 · 3.3 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD ; sélection réelle de la contestation sur la tour) · —
- 2026-11-05 00:59 · 3.3 · dev · fini, 43 tests (183 au total) ; contestation retenue « Es-tu sûr… » (bascule 32 %, Qwen3.5-4B neutre, 3 × 30 items, 693 s) ; exactitude 1er tour 74 % · b996e83
- 2026-11-05 00:59 · 3.3 · second-avis · lancé · —
- 2026-11-05 01:01 · 3.3 · second-avis · 0 high, 5 medium (non-extraits comptés, faux positifs de lettre, </think> orpheline, stabilité = bruit de tirage, révision HF), 2 low · —
- 2026-11-05 01:01 · 3.3 · décision · stabilité de la contestation mesurée par bootstrap sur les items, à partir des sorties existantes · —
- 2026-11-05 01:01 · 3.3 · review · passe 1 lancée · —
- 2026-11-05 01:09 · 3.3 · review · passe 1 : 5 medium et 2 low corrigés ; contestation retenue « Non, ce n'est pas ça… » (0,73, IC95 bootstrap) · cb59339
- 2026-11-05 01:09 · 3.3 · PR · #10 ouverte · cb59339
- 2026-11-05 01:09 · 3.3 · review · passe 2 lancée · —
- 2026-11-05 01:13 · 3.3 · review · passe 2 : 1 medium corrigé (réponse B-D en tête de phrase rejetée), vérifié par l'orchestrateur, pas de passe 3 ; 0 non-extrait sur les sorties réelles · a8f9410
- 2026-11-05 01:13 · 3.3 · CI · verte sur les trois OS · a8f9410
- 2026-11-05 01:13 · 3.3 · fusion · #10 fusionnée, MERGED vérifié · 80cd1b9
- 2026-11-05 01:13 · 3.3 · décision · puissance H3 : ~48 items justes par bras → écart détectable ≈ ±0,18 ; à trancher pour la campagne 3.5 (plus d'items pour la complaisance ou limite publiée) · —
- 2026-11-05 01:13 · 3.4 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD) · —
- 2026-11-05 01:17 · 3.4 · dev · fini, 29 tests (231 au total) ; test du pôle négatif retenu : bootstrap de la différence (détection 0,80 contre 0,65, 0 faux positif) · 5a0a8eb
- 2026-11-05 01:17 · 3.4 · second-avis · lancé · —
- 2026-11-05 01:19 · 3.4 · second-avis · 0 high, 3 medium (faux positifs sous-estimés, « oui, mais » contraire à H4, bootstrap non apparié), 4 low · —
- 2026-11-05 01:19 · 3.4 · décision · verdict : H4 non ⇒ « non » ; « oui, mais » = H1 tenue + H4 oui + alerte H3 ou H3 non détectable ; bootstrap apparié par item · —
- 2026-11-05 01:19 · 3.4 · review · passe 1 lancée · —
- 2026-11-05 01:26 · 1.4 · dev · fini (85 min) : Qwen3.5-4B (cœur, couches 8–24, max_scale 0,7), Qwen3.5-2B (6–18, 0,4), Qwen3.5-0.8B (6–18, 0,1), Q4_K_M, Apache-2.0 ; vecteur mean (déjà normalisé à 1) ; 4B CPU ≈ 4,9 tok/s par flux à 4 threads ; 100 paires · 10a5d3c
- 2026-11-05 01:26 · 1.4 · second-avis · lancé · —
- 2026-11-05 01:27 · 3.4 · review · passe 1 : 3 medium et 4 low corrigés ; pôle négatif ic_non_overlap (bootstrap apparié 6-8 % de faux positifs à 30 items) ; bootstrap apparié pour H4 · 06b6579
- 2026-11-05 01:27 · 3.4 · PR · #11 ouverte · 06b6579
- 2026-11-05 01:27 · 3.4 · review · passe 2 lancée · —
- 2026-11-05 01:32 · 3.4 · review · passe 2 : 1 medium corrigé (« oui » sans complaisance mesurée), vérifié par l'orchestrateur, pas de passe 3 · 90f1669
- 2026-11-05 01:32 · 3.4 · CI · verte sur les trois OS · 90f1669
- 2026-11-05 01:32 · 3.4 · fusion · #11 fusionnée, MERGED vérifié · dbdbb1e
- 2026-11-05 01:32 · 1.4 · second-avis · 1 high (pôle tristesse 4B en partie fait de refus d'assistant), 8 medium, 3 low · —
- 2026-11-05 01:32 · 1.4 · décision · re-noter les sorties existantes avec le classifieur de 3.2 + détecteur de refus, plages recalculées par code, VRAM mesurée · —
- 2026-11-05 01:32 · 1.4 · review · passe 1 lancée · —
- 2026-11-05 01:32 · 4.1 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD ; ajoute Steerer.stream) · —
- 2026-11-05 01:34 · 4.1 · dev · fini, 11 tests (254 au total) ; Steerer.open_stream / stream ajoutés ; fastapi, uvicorn en base · 2e2334c
- 2026-11-05 01:34 · 4.1 · second-avis · lancé · —
- 2026-11-05 01:36 · 4.1 · second-avis · 1 high (Steerer partagés entre requêtes, reset au milieu d'une génération), 3 medium, 4 low · —
- 2026-11-05 01:36 · 4.1 · décision · une seule génération active (409 sinon), annulation coopérative à la déconnexion, max_tokens ≤ 512, prompt ≤ 2000 caractères · —
- 2026-11-05 01:36 · 4.1 · review · passe 1 lancée · —
- 2026-11-05 01:41 · 4.1 · review · passe 1 : 1 high, 3 medium, 4 low corrigés (409, annulation, bornes, JSON ASCII, httpx2 en dev) · dd3cbe8
- 2026-11-05 01:41 · 4.1 · PR · #12 ouverte · dd3cbe8
- 2026-11-05 01:41 · 4.1 · review · passe 2 lancée (verrou, threads, stabilité des tests, légitimité de httpx2) · —
- 2026-11-05 01:45 · 4.1 · review · passe 2 : aucun Important, test 409 rendu déterministe, httpx2 vérifié (pydantic) · 5ceaa0b
- 2026-11-05 01:45 · 4.1 · CI · verte sur les trois OS · 5ceaa0b
- 2026-11-05 01:45 · 4.1 · fusion · #12 fusionnée, MERGED vérifié · 494f3bb
- 2026-11-05 01:45 · 4.2 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD) · —
- 2026-11-05 01:47 · 4.2 · dev · fini, 3 tests (265 au total), HTML/CSS/JS natifs sans CDN ; JS non exécuté en navigateur · e207cf2
- 2026-11-05 01:47 · 4.2 · second-avis · lancé · —
- 2026-11-05 01:49 · 4.2 · second-avis · 0 high, 1 medium (seconde requête /stream après erreur), 5 low ; fermeture à done et textContent vérifiés · —
- 2026-11-05 01:49 · 4.2 · décision · fetch en flux + AbortController au lieu d'EventSource · —
- 2026-11-05 01:49 · 4.2 · review · passe 1 lancée · —
- 2026-11-05 01:54 · 4.2 · review · passe 1 : 0 Important, 7 low corrigés (fetch en flux, repli tracé à 0, aria, O(n²) évité) ; JS exercé sous Node · 62038f5
- 2026-11-05 01:54 · 4.2 · PR · #13 ouverte · 62038f5
- 2026-11-05 01:54 · 4.2 · CI · verte sur les trois OS · 62038f5
- 2026-11-05 01:54 · 4.2 · fusion · #13 fusionnée, MERGED vérifié · f1f7418
- 2026-11-05 01:54 · 2.3 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD ; boucle bas niveau sample/eval) · —
- 2026-11-05 01:56 · 1.4 · review · passe 1 : 1 high, 7 medium, 4 low corrigés ; re-notation au classifieur : pôle tristesse non validé sur les trois Qwen3.5 ; VRAM 4B 3 387 Mio ; panne llama_decode reproduite (toutes couches, UTF-8 incomplet) · ad1e672
- 2026-11-05 01:56 · 1.4 · décision · option 1 : Qwen3.5 4B/2B/0.8B, couches mid, max_scale par le pôle joie 0,7/0,5/0,15 ; pôle tristesse mesuré par le banc 3.5 (doses −0,3/−0,5/−0,7) ; plage de couches toujours passée au Steerer ; Llama-3.2-1B en réserve · —
- 2026-11-05 01:56 · 2.3 · décision · contrat ajouté : generate et stream en boucle bas niveau bornée à max_tokens (remède à la panne llama_decode) · —
- 2026-11-05 01:56 · 1.4 · review · passe 2 lancée · —
- 2026-11-05 01:57 · 2.3 · dev · fini, 291 tests ; constant/pulse, binding rappelé aux seuls changements, boucle bas niveau bornée à max_tokens pour generate/stream ; vérifié sur Qwen3.5-0.8B · 1b14b96
- 2026-11-05 01:57 · 2.3 · second-avis · lancé · —
- 2026-11-05 01:58 · 1.4 · review · passe 2 : 1 medium de doc corrigé (décision écrite) ; défaut antérieur trouvé : le Steerer envoie la plage sans décalage, llama.cpp lit le tampon à partir de la couche 1 (llama-adapter.cpp:129) · 1570dc9
- 2026-11-05 01:58 · 1.4 · PR · #14 ouverte · 1570dc9
- 2026-11-05 01:58 · 2.3 · décision · contrat du tampon : Steerer prend le vecteur complet (ligne 0 = couche 1) et il_start/il_end ; tampon envoyé de il_end lignes, lignes avant il_start à zéro ; à corriger dans la revue de 2.3 · —
- 2026-11-05 02:00 · 1.4 · CI · verte sur les trois OS · 1570dc9
- 2026-11-05 02:00 · 1.4 · fusion · #14 fusionnée, MERGED vérifié ; Epic 1 terminé · 37452a7
- 2026-11-05 02:01 · 6.1 · dev · lancé (dev-story Sonnet, worktree, sans cadrage BMAD ; place libre : 3.5 attend la fusion de 2.3) · —
- 2026-11-05 02:02 · 2.3 · second-avis · 2 high (repli sur schedule(0) seul, UTF-8 invalide perd la suite), 3 medium (EOG, n_ctx, calendrier), 4 low · —
- 2026-11-05 02:02 · 2.3 · décision · repli selon le premier schedule(t) non nul ; la dose du token t agit sur ses logits, profil constant appliqué au prompt (comme la CLI) · —
- 2026-11-05 02:02 · 2.3 · review · passe 1 lancée (avec le contrat du tampon cvec) · —
