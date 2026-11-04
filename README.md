# mood-steering-slm

Donner une « humeur » (joie, calme…) à un petit modèle de langage sans toucher à ses poids : un vecteur de steering ajouté aux activations au moment de l'inférence, dosable, réversible et comparé à un simple prompt de jeu de rôle.

> **État : spécification.** Aucun code pour l'instant. Le point de départ est dans [`docs/brief-steering-humeur.md`](docs/brief-steering-humeur.md) ; le cadrage BMAD suit dans `_bmad-output/planning-artifacts/`.

## Idée

Une molécule comme l'ocytocine change le comportement d'un cerveau sans le reconstruire. Pour un LLM, l'équivalent le plus proche est le **steering d'activations** ([ActAdd](https://arxiv.org/abs/2308.10248), [Representation Engineering](https://arxiv.org/abs/2310.01405), [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude)). On calibre une direction « joie » à partir d'une centaine de paires de prompts contrastées, sans gradient. On l'ajoute ensuite au flux résiduel avec une intensité réglable, qui peut même décroître au fil des tokens, comme une « pharmacocinétique ».

Le démonstrateur doit répondre à des questions mesurables :

- l'effet est-il net avant que le texte ne se dégrade, et pour quelle plage d'intensité ?
- que coûte-t-il en exactitude et en complaisance ?
- qu'apporte-t-il face à un system prompt « sois joyeux » ?
- quel surcoût en latence et en énergie ?

## Ce que ce n'est pas

Le démonstrateur montre l'**expression** d'une humeur dans le comportement du modèle. Il ne dit rien d'un éventuel état vécu.

## Projets liés

- [slm-weight-editing](https://github.com/Aliquanto3/slm-weight-editing) : l'autre métaphore, la « chirurgie », qui inscrit une direction directement dans les poids, de façon permanente.
- [slm-quality-bench](https://github.com/Aliquanto3/slm-quality-bench) : banc d'évaluation en français des SLM.
- [frugal-ai-workbench](https://github.com/Aliquanto3/frugal-ai-workbench) : workbench SLM local, intégration possible à terme.

## Licence

[Apache 2.0](LICENSE).
