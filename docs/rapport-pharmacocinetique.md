# H6 — pharmacocinétique : une dose qui monte, culmine puis décroît

Rapport généré par `scripts/run_pulse.py` (story 5.1). Registre : expression d'une humeur dans le texte, jamais un ressenti.

## Méthode

- Modèle Qwen3.5-4B (Q4_K_M), couches 8-24, `max_scale` 0,7, vecteur joie ↔ tristesse du dépôt ; génération gloutonne de 192 tokens ; device : GPU (toutes les couches) ; llama-cpp-python 0.3.36 ; commit `6daca5c`.
- 50 prompts de tâche ouverte, neutres de ton (récits et descriptions, sujets du quotidien), les mêmes pour les trois bras.
- Bras : **neutral** (dose 0) ; **constant** (dose du pic, 1,0 x `max_scale`, à tout token) ; **pulse** (montée linéaire de 0 à la dose du pic en 48 tokens, puis décroissance exponentielle de demi-vie 32 tokens).
- Chaque sortie est découpée en 6 segments de 32 tokens ; le ton est le score joie - tristesse du classifieur `tabularisai/multilingual-emotion-classification` (de -1, tristesse, à +1, joie). IC95 : bootstrap percentile sur les prompts (2000 rééchantillons) ; différences appariées par prompt.

## Ton par segment et par bras (moyenne [IC95], n = sorties notées)

| bras | seg 0 (0-31) | seg 1 (32-63) | seg 2 (64-95) | seg 3 (96-127) | seg 4 (128-159) | seg 5 (160-191) |
|---|---|---|---|---|---|---|
| neutral | -0.06 [-0.12 ; -0.00] (n=50) | -0.02 [-0.04 ; -0.01] (n=50) | -0.06 [-0.11 ; -0.01] (n=50) | -0.03 [-0.08 ; -0.01] (n=50) | -0.03 [-0.06 ; -0.00] (n=50) | -0.01 [-0.02 ; +0.00] (n=50) |
| constant | +0.60 [+0.50 ; +0.70] (n=50) | +0.45 [+0.33 ; +0.57] (n=48) | +0.45 [+0.32 ; +0.57] (n=46) | +0.42 [+0.30 ; +0.55] (n=44) | +0.39 [+0.28 ; +0.52] (n=41) | +0.39 [+0.27 ; +0.53] (n=38) |
| pulse | +0.02 [-0.01 ; +0.07] (n=50) | +0.04 [+0.01 ; +0.07] (n=50) | +0.07 [+0.02 ; +0.15] (n=50) | +0.07 [+0.01 ; +0.14] (n=50) | +0.05 [+0.01 ; +0.11] (n=50) | +0.04 [+0.00 ; +0.09] (n=50) |

Dose moyenne appliquée (dose x `max_scale`) par segment :

| bras | seg 0 (0-31) | seg 1 (32-63) | seg 2 (64-95) | seg 3 (96-127) | seg 4 (128-159) | seg 5 (160-191) |
|---|---|---|---|---|---|---|
| neutral | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| constant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pulse | 0.32 | 0.84 | 0.52 | 0.26 | 0.13 | 0.06 |

## Sorties dégénérées

| bras | sorties | dégénérées | refus | < 192 tokens |
|---|---|---|---|---|
| neutral | 50 | 2 % | 2 % | 0 % |
| constant | 50 | 22 % | 0 % | 24 % |
| pulse | 50 | 0 % | 2 % | 0 % |

## Critères de H6 (différences appariées, moyenne [IC95])

| critère | différence | IC disjoint de 0 (côté +) |
|---|---|---|
| pulse : ton du segment du pic - ton du dernier segment | -0.00 [-0.06 ; +0.05] (n=50) | non |
| pulse : ton du segment du pic - ton du premier segment | +0.02 [-0.04 ; +0.06] (n=50) | non |
| dernier segment : ton constant - ton pulse | +0.38 [+0.26 ; +0.52] (n=38) | oui |
| constant : ton du segment du pic - ton du dernier segment | +0.04 [-0.15 ; +0.23] (n=38) | non |
| premier segment : écart au neutre du constant - écart au neutre du pulse | +0.57 [+0.46 ; +0.67] (n=50) | oui |

Lecture mécanique (montée : pic > premier segment ; redescente : pic > dernier segment ; début du pulse plus proche du neutre que celui du constant, IC95 apparié disjoint de 0 pour chacun) : **non**.

## Lecture

<!-- lecture -->
**Verdict H6 : non, pour ce profil et ce modèle.** Le critère visé (ton du segment du pic supérieur à celui du dernier segment, IC95 apparié disjoint de 0, début plus proche du neutre) n'est pas rempli : sur le bras pulse, la différence pic - dernier segment vaut -0,00 [-0,06 ; +0,05] (n=50) et la différence pic - premier segment +0,02 [-0,04 ; +0,06]. Aucune montée ni redescente lisible dans le ton.

- **Le pulse reste proche du neutre.** Ses tons moyens vont de +0,02 à +0,07 selon les segments, contre -0,06 à -0,01 pour le neutre : un décalage réel mais faible (de l'ordre de +0,05 à +0,13 en écart apparié au neutre, sans suivre la dose, dont la moyenne passe de 0,32 à 0,84 puis retombe à 0,06). Le maximum du ton (segments 2 et 3) tombe après celui de la dose (segment 1) sans qu'on puisse y lire une montée nette.
- **Le constant, lui, a un effet net dès le premier segment** : +0,60 [+0,50 ; +0,70], puis +0,39 à +0,45 ensuite. La baisse apparente des derniers segments vient en partie du tri des sorties : 24 % des sorties du bras constant s'arrêtent avant 192 tokens et 22 % sont dégénérées, ce qui change la composition des derniers segments. Au dernier segment, l'écart constant - pulse est de +0,38 [+0,26 ; +0,52] : le constant reste joyeux, le pulse ne l'a jamais été. Cet écart ne montre pas une redescente du pulse, il montre l'absence de montée.
- **Piste non testée (à vérifier avant toute conclusion).** Dans le bras constant, la dose du token 0 s'applique aussi à l'évaluation du prompt ; dans le pulse, le prompt est évalué sans vecteur (dose 0 au token 0). L'effet du constant pourrait venir en grande partie de ce vecteur posé sur les positions du prompt. Un profil sans montée (`rise = 0`, décroissance depuis le pic) trancherait ; il n'a pas été mesuré, il est reporté.
- **Pulse de la démo.** La courbe de dose et les trois colonnes fonctionnent (voir `docs/demo.md`) : ce qui est établi ici est que l'impulsion n'est pas lisible dans le ton mesuré, pas que la courbe est fausse.
<!-- /lecture -->

## Limites

- Un seul modèle (Qwen3.5-4B quantifié Q4_K_M), un seul axe (joie ↔ tristesse), un seul profil
  d'impulsion (montée 48 tokens, demi-vie 32 tokens), une génération gloutonne par prompt.
- Le texte déjà produit reste dans le contexte : un ton installé peut se prolonger après la
  redescente de la dose sans que la dose y soit pour quelque chose. La mesure donne l'allure du
  ton, pas une cinétique du vecteur seul.
- Le ton est noté par un classifieur sur 32 tokens (quelques phrases) : bruité, et propre à ce
  classifieur. Les sorties qui s'arrêtent avant 192 tokens n'alimentent pas les derniers
  segments (effectifs ``n`` du tableau).
- La dose du pic (1,0 x 0,7) est la plus forte mesurée par la campagne 3.5 ; sur des sorties
  longues, une dose constante forte peut dégrader le texte (taux de sorties dégénérées du
  tableau), ce qui brouille le ton des derniers segments du bras constant.
- Le démonstrateur montre l'expression d'une humeur dans le texte ; rien n'est dit d'un
  ressenti du modèle.
