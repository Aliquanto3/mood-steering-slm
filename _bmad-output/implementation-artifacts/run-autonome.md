# Run autonome — plan et état

Fichier d'état de l'orchestrateur, à relire en premier après chaque `/clear`. Lu par `/agent-cost-kit:avancement-run`. Machine du run : la tour (RTX 3050, 8 Go), qui exécute aussi les pics et la campagne.

## Périmètre

| Périmètre | Epics | Note |
|---|---|---|
| Dans le run | Epics 1 à 4 | MVP : pics, bibliothèque, banc et verdict, démonstrateur |
| Hors run | Epics 5 à 6 | bonus (mesures complémentaires, anti-boucle), après le MVP |

Au plus 2 stories en développement à la fois, chacune dans son worktree.
Plafond global : 5 agents actifs à la fois sous Windows (développement, revue et rebase confondus).

## Règles

- Types : `dev` (dev-story → second-avis → reviewer → intégration), `mixte` (code par l'agent, puis « attente humaine »), `humaine` (jamais confiée à un agent). ★ = story pivot, dev-story sur Opus.
- Petites stories bien décrites dans `epics.md` : dev-story directement sur l'entrée d'epics, sans fichier de story BMAD (règle expérimentale d'aiguillage, effet noté au journal).
- Deux stories du même fichier partagé ne vont jamais dans la même vague, sauf `pyproject.toml` / `uv.lock` (ajout de dépendance) : la seconde PR à fusionner rebase et relance `uv lock`.
- Fichiers d'état du run (ce fichier, `sprint-status.yaml`) : jamais dans une PR de story. PR `chore/etat-run` à part.
- Suite complète : pas en local pendant un run parallèle, la CI de branche la joue et sert de porte de fusion. Attente : `gh run watch <id> --interval 60`.
- Fusion par merge commit (`gh pr merge --merge`), jamais de push sur `main`. Journaliser `fusion` seulement après avoir vérifié `MERGED`, puis `git log -1` sur `main`.
- `git worktree prune` avant chaque lancement. Les worktrees ne sont supprimés qu'en fin de run.
- Heure du journal : toujours prise par `date` au moment d'écrire, jamais devinée. `uv run python` sous Windows.
- Ressources humaines : `ANTHROPIC_API_KEY` absente au lancement (juge de 3.2 simulé en test ; appel réel en 3.5) ; test sur le portable (4.3), non bloquant. Prévenir Anaël par notification et avancer sur le reste.

## Sessions autorisées à modifier le périmètre

- (aucune)

## Contrats partagés en cours

- Paquet `mood_steering` sous `src/` (layout src), Python ≥ 3.11, version lue par `mood_steering.__version__`.
- `llama-cpp-python` en extra optionnel `llama` (`uv add --optional llama llama-cpp-python`) : la CI ne l'installe pas, les tests simulent le binding. `gguf` en dépendance de base (pur Python).
- Extra optionnel `eval` (transformers, torch, anthropic) pour 3.2 ; le cœur ne l'importe pas.
- Modèles téléchargés dans `models/`, vecteurs dans `data/moods/{model_id}/{axe}.gguf`, sorties du banc dans `data/runs/` : jamais versionnés.

## Sprint 1 — Fondations et pics (Epic 1)

| Vague | Stories en parallèle | Après | Fichiers ou écrans touchés |
|---|---|---|---|
| V0 | 1.1 ★ | — | `pyproject.toml`, `src/mood_steering/__init__.py`, `.github/workflows/ci.yml`, `.gitignore` |
| V1 | 1.2 (mixte) · 1.3 (mixte) | 1.2 ← 1.1 ; 1.3 ← 1.1 | 1.2 `spikes/cvec_between_tokens.py`, `docs/pics/cvec-entre-tokens.md` ; 1.3 `spikes/read_cvector_gguf.py`, `docs/pics/lecture-gguf-cvector.md` ; les deux `pyproject.toml` |
| V2 | 1.4 ★ (mixte) | 1.4 ← 1.2, 1.3 | `spikes/calibration_scan.py`, `src/mood_steering/pairs/joy_sadness.jsonl`, `docs/pics/choix-des-modeles.md` |

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

« Après » ne lit que `x ← y` ou `Epics N à M`.

## Journal

Une ligne par étape : `- AAAA-MM-JJ HH:MM · story · étape · résultat · commit`. Une story est terminée quand sa PR est fusionnée.

- 2026-11-04 23:24 · — · décision · sprint planning : gate PASS, 6 epics, 21 stories, run Epics 1 à 4, 2 stories en parallèle au plus · —
- 2026-11-04 23:26 · 1.1 · dev · lancé (dev-story Opus, worktree, sans cadrage BMAD) · —
- 2026-11-04 23:27 · 1.1 · dev · fini, 2 tests verts, ruff et contrôle rapide OK · cc89529
