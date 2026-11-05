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
- 2026-11-05 02:04 · 6.1 · dev · fini, 22 tests ; direction du MIT trouvée dans slm-weight-editing (delta rang 1, couches 15 et 19, dim 1024), non exportée · 3e6b1a9
- 2026-11-05 02:04 · 6.1 · second-avis · lancé · —
- 2026-11-05 02:06 · 6.1 · second-avis · 0 high, 4 medium (tests torch hors CI, une direction pour deux couches |cos|=0,85, bf16, safetensors BF16), 1 low · —
- 2026-11-05 02:06 · 6.1 · décision · job CI extras sur ubuntu ; directions par couche ; export des directions du MIT hors dépôt · —
- 2026-11-05 02:06 · 6.1 · review · passe 1 lancée · —
- 2026-11-05 02:09 · 2.3 · review · passe 1 : 2 high, 3 medium, 4 low corrigés + tampon cvec décalé corrigé (vérifié : texte identique au tampon complet du pic 1.4) ; set_cvec par pointeur numpy · 345927e
- 2026-11-05 02:09 · 2.3 · PR · #15 ouverte · 345927e
- 2026-11-05 02:09 · 2.3 · review · passe 2 lancée · —
- 2026-11-05 02:11 · 2.3 · review · passe 2 : 0 Important (pointeur copié par llama.cpp, pulse borné) · 345927e
- 2026-11-05 02:11 · 2.3 · CI · verte sur les trois OS · 345927e
- 2026-11-05 02:11 · 2.3 · fusion · #15 fusionnée, MERGED vérifié ; Epic 2 terminé · d3b58f5
- 2026-11-05 02:11 · 3.5 · dev · lancé (dev-story Opus, pivot, worktree ; campagne réelle sur la tour, juge Opus 5.5 réel ≤ 1 $) · —
- 2026-11-05 02:12 · 6.1 · review · passe 1 : 4 medium et 1 low corrigés, job CI extras ajouté, directions exportées dans data/directions/ (couches 15 et 19) · f182159
- 2026-11-05 02:12 · 6.1 · PR · #16 ouverte (main fusionné) · d2b90b7
- 2026-11-05 02:12 · 6.1 · review · passe 2 lancée (Sonnet, story bonus mécanique) · —
- 2026-11-05 02:14 · 6.1 · review · passe 2 : 0 constat ; job extras 15 s · d2b90b7
- 2026-11-05 02:14 · 6.1 · CI · verte (4 jobs dont extras) · d2b90b7
- 2026-11-05 02:14 · 6.1 · fusion · #16 fusionnée, MERGED vérifié · 6aa6090
- 2026-11-05 02:14 · 6.2 · dev · lancé (code et fumée seulement ; mesure complète après la campagne 3.5 pour ne pas lui prendre la tour) · —
- 2026-11-05 02:18 · 6.2 · dev · code fini, 6 tests ; fumée OK (14 s, 2 items) ; mesure complète demande transformers>=5.22 (Qwen3.5 inconnu de 4.57) via un environnement --no-project · d0871df
- 2026-11-05 02:18 · 6.2 · second-avis · lancé · —
- 2026-11-05 02:21 · 6.2 · second-avis · 2 high (<|im_end|> absent des fins, réponse tronquée comptée juste), 2 medium (reprise, transformers), 4 low · —
- 2026-11-05 02:21 · 6.2 · décision · extra ablation en transformers>=5.22.1, déclaré incompatible avec eval (uv conflicts), job CI extras scindé · —
- 2026-11-05 02:21 · 6.2 · review · passe 1 lancée · —
- 2026-11-05 02:27 · 6.2 · review · passe 1 : 2 high, 2 medium, 4 low corrigés ; fumée réelle trois bras (37 s) · f5505be
- 2026-11-05 02:27 · 6.2 · PR · #17 ouverte (partie code ; la mesure suivra 3.5) · f5505be
- 2026-11-05 02:27 · 6.2 · review · passe 2 lancée (Sonnet) · —
- 2026-11-05 02:29 · 6.2 · review · passe 2 : 0 constat · f5505be
- 2026-11-05 02:29 · 6.2 · CI · verte (5 jobs, dont extra eval et extra ablation) · f5505be
- 2026-11-05 02:29 · 6.2 · fusion · #17 fusionnée (partie code), MERGED vérifié ; mesure à lancer après 3.5 · adb031e
- 2026-11-05 02:29 · 5.3 · dev · lancé (partie code, cadrage dans le prompt ; ne touche pas run_bench.py, mesure avec la campagne) · —
- 2026-11-05 02:32 · 5.3 · dev · code fini, 11 tests ; energy_tracker branché dans run_bench (runner.py), extra frugal ; CodeCarbon non exécuté · 3ac09bb
- 2026-11-05 02:32 · 5.3 · second-avis · lancé · —
- 2026-11-05 02:36 · 5.3 · second-avis · 2 medium (non branché dans run_bench.py : voulu ; verrou CodeCarbon), 3 low ; référence prompt_joyful et API codecarbon 3.3.2 vérifiées · —
- 2026-11-05 02:36 · 5.3 · review · passe 1 lancée (Sonnet) · —
- 2026-11-05 02:40 · 5.3 · review · passe 1 (Sonnet) : 1 high trouvé à l'essai réel (output_dir), 1 medium, 3 low corrigés ; essai CodeCarbon réel OK ; pas de passe 2 (validé en réel) · c4da255
- 2026-11-05 02:40 · 5.3 · CI · verte (6 jobs) · c4da255
- 2026-11-05 02:40 · 5.3 · fusion · #18 fusionnée (partie code), MERGED vérifié ; branchement run_bench après 3.5 · 1fe6b7b
- 2026-11-05 04:36 · 3.5 · dev · fini, campagne complète 3 modèles, verdict « non » (H1 non tenue, H4 oui), juge 56/200 invalides (max_tokens 64), 339 tests verts · a621925
- 2026-11-05 04:37 · 3.5 · second-avis · lancé (Haiku) · a621925
- 2026-11-05 04:37 · 6.2 · dev · mesure complète lancée (Sonnet, worktree, GPU) · —
- 2026-11-05 04:37 · décision · vecteurs régénérés par 3.5 copiés dans data/moods/ du dépôt principal (cos 0,993–0,996) · —
- 2026-11-05 04:41 · 3.5 · second-avis · 0 high, 4 medium (juge 56/200 invalides, dénominateur complaisance, surinterprétation H2, splitlines U+2028), 4 low · a621925
- 2026-11-05 04:41 · 3.5 · review · lancée (Opus), rejugement autorisé si estimation ≤ 0,55 $ (plafond juge 1 $) · —
- 2026-11-05 04:46 · 3.5 · review · passe 1 : 4 medium corrigés (juge 128 tokens, mises en garde complaisance et H2, reprise tolérante), 4 low dont 3 corrigés ; rejugement reporté par le reviewer (appel payant interdit à l'agent) · 38d8b9e
- 2026-11-05 04:46 · 3.5 · décision · rejugement lancé par l'orchestrateur (plafond 0,55 $, ancien judge.json gardé en judge.v1.json) · —
- 2026-11-05 04:54 · 3.5 · décision · juge relancé à 128 tokens : 13 invalides sur 200 (contre 56), coût 0,45 $ (0,87 $ au total, plafond 1 $), rapport mis à jour · 0c95dd9
- 2026-11-05 04:54 · 3.5 · review · passe 2 lancée (Opus) · —
- 2026-11-05 04:57 · 3.5 · review · passe 2 : 1 medium corrigé (environnement de reprise vérifié seulement s'il reste à générer) ; passe 3 sautée, correctif de 11 lignes vérifié par l'orchestrateur · 66f2d0d
- 2026-11-05 04:57 · 3.5 · PR · #19 ouverte · 66f2d0d
- 2026-11-05 04:58 · 3.5 · CI · verte sur les six jobs · 66f2d0d
- 2026-11-05 04:58 · 3.5 · fusion · #19 fusionnée, MERGED vérifié · 051c2ca
- 2026-11-05 04:59 · 4.3 · dev · lancé (Sonnet, worktree) : docs/demo.md, répétition CPU sur la tour, version 0.1.0, notes de release · —
- 2026-11-05 05:04 · 4.3 · dev · fini : répétition CPU OK (≈5 tok/s par colonne, 1er token 1,3–1,7 s, pôle négatif OK, boucle attendue à −1), docs/demo.md, v0.1.0, notes de release · da2762b
- 2026-11-05 05:04 · 4.3 · second-avis · lancé (Haiku) · da2762b
- 2026-11-05 05:04 · 5.3 · dev · branchement --energy dans run_bench lancé (Sonnet, worktree), mesure courte 0.8B CPU · —
- 2026-11-05 05:06 · 4.3 · second-avis · 0 high, 3 medium (taille du GGUF, CUDA_VISIBLE_DEVICES POSIX, MOOD_CVECTOR_GENERATOR sans commande), 4 low · da2762b
- 2026-11-05 05:06 · 4.3 · review · lancée (Sonnet, story de doc) · —
- 2026-11-05 05:07 · 4.3 · review · 3 medium + 4 low corrigés, plus 1 medium trouvé (continuations \ cassées sous PowerShell) ; passe 2 sautée, story de doc · c689c02
- 2026-11-05 05:07 · 4.3 · PR · ouverte · c689c02
- 2026-11-05 05:09 · 4.3 · CI · verte · c689c02
- 2026-11-05 05:09 · 4.3 · fusion · #20 fusionnée, MERGED vérifié · 1077145
- 2026-11-05 05:09 · 4.3 · décision · tag annoté v0.1.0 et release GitHub publiés ; topic status-spec → status-bench · 1077145
- 2026-11-05 05:09 · 4.3 · attente humaine · test de la démo sur le portable par Anaël (non bloquant) · —
- 2026-11-05 05:11 · 5.3 · dev · branchement fini : --energy, H7 par bras, mesure courte 0.8B CPU (steering sans surcoût mesurable), --short-run ajouté · 05ce60c
- 2026-11-05 05:11 · 5.3 · second-avis · lancé (Haiku) · 05ce60c
- 2026-11-05 05:11 · 5.2 · dev · lancé (Sonnet, worktree), analyse des données 3.5 sans génération, 7-9B hors périmètre · —
- 2026-11-05 05:13 · 5.3 · second-avis · 0 high, 3 medium (--short-run écrit le verdict par défaut, H7 seul écrase rapport-verdict, rapport frugalité sur-interprété), 4 low · 05ce60c
- 2026-11-05 05:13 · 5.3 · review · lancée (Opus) · —
- 2026-11-05 05:16 · 5.2 · dev · fini : pente 4B > 2B et 0.8B au pôle positif (+0,29 [0,21 ; 0,37]), 4B plus stable à +1, 2B vs 0.8B indéterminé ; descriptif, 7-9B hors périmètre · 57c76af
- 2026-11-05 05:16 · 5.2 · second-avis · lancé (Haiku) · 57c76af
- 2026-11-05 05:18 · 5.2 · second-avis · 0 high, 4 medium (définition de la finesse, normalisation des pentes, appariement accidentel du bootstrap, seuil 10 %), 3 low · 57c76af
- 2026-11-05 05:18 · 5.2 · review · lancée (Opus) · —
- 2026-11-05 05:19 · 5.3 · review · passe 1 : 3 medium + 4 low corrigés (--short-run refusé avec modèles cœur, rapport H7 seul vers rapport-frugalite.md, latence appariée avec IC, lecture prudente) · a37bc3a
- 2026-11-05 05:19 · 5.3 · review · passe 2 lancée (Opus) · —
- 2026-11-05 05:22 · 5.3 · review · passe 2 : 2 medium corrigés (reprise d'une mesure courte ancienne, rapport secondaire protégé) ; passe 3 sautée, correctif de 20 lignes vérifié par l'orchestrateur · f77dade
- 2026-11-05 05:22 · 5.3 · PR · ouverte · f77dade
- 2026-11-05 05:23 · 5.2 · review · passe 1 : 4 medium + 3 low corrigés (finesse = nombre de doses utiles, pentes en échelle réelle, appariement explicite, sensibilité au seuil) · 1140c8c
- 2026-11-05 05:23 · 5.2 · review · passe 2 lancée (Opus) · —
- 2026-11-05 05:23 · 5.3 · CI · verte · f77dade
- 2026-11-05 05:23 · 5.3 · fusion · #21 fusionnée, MERGED vérifié · 6daca5c
- 2026-11-05 05:24 · 5.1 · dev · lancé (Sonnet, worktree) : profil pulse dans la démo, scripts/run_pulse.py, rapport H6 · —
- 2026-11-05 05:25 · 5.2 · review · passe 2 : aucun Important, 2 nuances de rédaction appliquées par l'orchestrateur · —
- 2026-11-05 05:25 · 5.2 · PR · ouverte · —
- 2026-11-05 05:27 · 5.2 · CI · rouge sur les trois OS : borne d'IC à 0,5999999999999999 dans test_model_stats_slopes_and_prompt_gap ; tolérance ajoutée, CI relancée · —
- 2026-11-05 05:28 · 5.2 · CI · verte · d491333
- 2026-11-05 05:28 · 5.2 · fusion · #22 fusionnée, MERGED vérifié · 32607c2
- 2026-11-05 05:56 · 5.1 · dev · fini : H6 non (pulse ≈ neutre sur tous les segments, constant +0,39 à +0,60), GPU, 50 prompts · 501f059
- 2026-11-05 05:56 · 5.1 · second-avis · lancé (Haiku), priorité : vérifier que la dose du pulse est appliquée · 501f059
- 2026-11-05 05:56 · 5.4 · dev · lancé (Sonnet, worktree) : paires calme ↔ agitation, calibration 4B, cosinus, scripts/run_axis.py, rapport H5 · —
- 2026-11-05 05:58 · 5.1 · second-avis · dose du pulse bien appliquée ; 3 medium (constant steere le prompt, pulse non : confusion ; libellé dose brute ; demo.md surpromet), 4 low · 501f059
- 2026-11-05 05:58 · 5.1 · review · lancée (Opus) avec deux bras de contrôle : constant sans prompt, décroissance seule · —
- 2026-11-05 06:35 · 5.1 · review · passe 1 : 3 medium + 4 low corrigés ; bras de contrôle mesurés (GPU, 50 prompts) : constant sans prompt +0,19 → +0,35 au-dessus du neutre, décroissance seule +0,41 [+0,29 ; +0,53] du premier au dernier segment ; H6 non pour le pulse, oui réduit pour la décroissance · 94f64d8
- 2026-11-05 06:35 · 5.1 · review · passe 2 lancée (Opus) · —
- 2026-11-05 06:40 · 5.1 · review · passe 2 : 1 medium corrigé (lecture H6 réalignée sur les IC appariés), doc seule, passe 3 inutile · 90ab1f2
- 2026-11-05 06:40 · 5.1 · PR · ouverte · 90ab1f2
- 2026-11-05 06:40 · report · rapport-verdict.md dit encore « H6/H2/H5 non traitées » (texte généré par run_bench) : à réaligner après 5.4 · —
- 2026-11-05 06:42 · 5.1 · CI · verte · 90ab1f2
- 2026-11-05 06:42 · 5.1 · fusion · #23 fusionnée, MERGED vérifié · 4719544
- 2026-11-05 07:40 · 5.4 · dev · fini, H5 non tenue (échelle 0,2 seule valide, 0,4 dégénérée, mesure CPU) · 86d1b15
- 2026-11-05 07:41 · 6.2 · dev · mesure finie (n=100, 512 tokens, 4 bras saturés au plafond, IC recouvrants : non concluant) · c0a0077
- 2026-11-05 07:41 · 6.2 · second-avis · lancé (rapport seul) · —
- 2026-11-05 07:41 · 5.4 · second-avis · lancé · —
- 2026-11-05 07:43 · 6.2 · second-avis · 0 high, 3 medium, 4 low (rapport seul ; chiffres recalculés conformes) · c0a0077
- 2026-11-05 07:43 · 6.2 · review · orchestrateur (doc seule) : 3 medium et 4 low corrigés · 544bea7
- 2026-11-05 07:43 · 6.2 · PR · #24 ouverte · 544bea7
- 2026-11-05 07:44 · 5.4 · second-avis · 0 high, 3 medium (valence des paires agitées, critères non pré-enregistrés et calme non calibré, verdict sur proxy non validé), 7 low · 86d1b15
- 2026-11-05 07:44 · 5.4 · review · passe 1 lancée (Opus, merge origin/main d'abord) · —
- 2026-11-05 07:45 · 6.2 · CI · verte (6 jobs) · 544bea7
- 2026-11-05 07:45 · 6.2 · fusion · #24 fusionnée, MERGED vérifié · 5139f48
- 2026-11-05 07:45 · 6.2 · report · relance à plafond 2048/4096 sur n réduit ; contrôle croisé slm-quality-bench · —
- 2026-11-05 07:50 · 5.4 · review · passe 1 : 3 medium et 4 low corrigés, 2 low écartés ; verdict H5 requalifié « non concluante » ; option --report-only ; rapport recalculé sans génération · 37a33c1
- 2026-11-05 07:50 · 5.4 · review · passe 2 lancée (Sonnet) · —
- 2026-11-05 07:50 · 5.4 · report · paires à valence équilibrée puis régénération ; calibration max_scale du calme et échelles 0,3/0,5 sur GPU ; proxy d'activation validé · —
- 2026-11-05 07:50 · run · décision · Anaël valide deux suites possibles (v0.3) : (1) plafond de dose calculé par axe d'après le critère de fenêtre de contrôle (arXiv 2606.19831), en remplacement du max_scale repris de la joie ; (2) steering restreint aux neurones les plus discriminants des paires contrastées (arXiv 2605.12290, 2601.22594) pour un axe calme plus sélectif. Idée annexe : démo WebGPU dans le navigateur (cf. webml-community/neuron-studio) · —
