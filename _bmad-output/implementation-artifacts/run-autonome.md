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
