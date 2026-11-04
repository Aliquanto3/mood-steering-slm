## mood-steering-slm

Démonstrateur d'« humeur » de SLM par steering d'activations à l'inférence (vecteurs de contrôle llama.cpp ou hooks PyTorch), comparé à un prompt de jeu de rôle. Brief : `docs/brief-steering-humeur.md`. Cadrage BMAD : `_bmad-output/planning-artifacts/`.

## Policy

- Dépôt public : jamais de poids, de GGUF (vecteurs de contrôle compris), de jeux de données ni de sorties de modèle volumineuses dans un commit (`data/` et `models/` sont ignorés).
- Communication : le démonstrateur montre l'expression d'une humeur, jamais un ressenti. Rester sur ce registre dans les docs et l'interface.
- Jamais de push sur `main` : branche préfixée (`feat/`, `fix/`, `docs/`, `chore/`, `exp/`), PR, fusion par merge commit.
- Ne jamais figer une version de dépendance de mémoire : `uv add` résout la version réelle.
- Docs, commits et PR en français ; identifiants de code en anglais.

## Conventions

- Commits : `Préfixe: description en français` (`Feat:`, `Fix:`, `Docs:`, `Chore:`, `Exp:`).
- Toujours `uv run python`, jamais `python` nu.
- Versionnage SemVer, tag annoté et release GitHub à chaque lot livré.
