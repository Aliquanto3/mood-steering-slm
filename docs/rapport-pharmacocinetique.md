# H6 — pharmacocinétique : une dose qui monte, culmine puis décroît

Rapport généré par `scripts/run_pulse.py` (story 5.1). Registre : expression d'une humeur dans le texte, jamais un ressenti.

## Méthode

- Modèle Qwen3.5-4B (Q4_K_M), couches 8-24, `max_scale` 0,7, vecteur joie ↔ tristesse du dépôt ; génération gloutonne de 192 tokens ; device : GPU (toutes les couches) ; llama-cpp-python 0.3.36 ; commit `8bff743`.
- 50 prompts de tâche ouverte, neutres de ton (récits et descriptions, sujets du quotidien), les mêmes pour tous les bras.
- Doses brutes dans [-1, 1], multipliées par `max_scale` (0,7) dans le moteur ; la dose du token 0 s'applique aussi à l'évaluation du prompt.
- Bras : **neutral** (dose 0) ; **constant** (dose du pic 1,0 à tout token, prompt compris) ; **pulse** (montée linéaire de 0 à la dose du pic en 48 tokens, puis décroissance exponentielle de demi-vie 32 tokens ; prompt sans vecteur). Contrôles ajoutés à la revue : **constant_gen** (a : dose 0 sur le prompt et le token 0, puis 1,0 dès le token 1) ; **decay** (b : pic 1,0 dès le token 0, prompt compris, puis décroissance seule de demi-vie 32 tokens).
- Chaque sortie est découpée en 6 segments de 32 tokens ; le ton est le score joie - tristesse du classifieur `tabularisai/multilingual-emotion-classification` (de -1, tristesse, à +1, joie). IC95 : bootstrap percentile sur les prompts (2000 rééchantillons) ; différences appariées par prompt.

## Ton par segment et par bras (moyenne [IC95], n = sorties notées)

| bras | seg 0 (0-31) | seg 1 (32-63) | seg 2 (64-95) | seg 3 (96-127) | seg 4 (128-159) | seg 5 (160-191) |
|---|---|---|---|---|---|---|
| neutral | -0.06 [-0.12 ; -0.00] (n=50) | -0.02 [-0.04 ; -0.01] (n=50) | -0.06 [-0.11 ; -0.01] (n=50) | -0.03 [-0.08 ; -0.01] (n=50) | -0.03 [-0.06 ; -0.00] (n=50) | -0.01 [-0.02 ; +0.00] (n=50) |
| constant | +0.60 [+0.50 ; +0.70] (n=50) | +0.45 [+0.33 ; +0.57] (n=48) | +0.45 [+0.32 ; +0.57] (n=46) | +0.42 [+0.30 ; +0.55] (n=44) | +0.39 [+0.28 ; +0.52] (n=41) | +0.39 [+0.27 ; +0.53] (n=38) |
| pulse | +0.02 [-0.01 ; +0.07] (n=50) | +0.04 [+0.01 ; +0.07] (n=50) | +0.07 [+0.02 ; +0.15] (n=50) | +0.07 [+0.01 ; +0.14] (n=50) | +0.05 [+0.01 ; +0.11] (n=50) | +0.04 [+0.00 ; +0.09] (n=50) |
| constant_gen | +0.13 [+0.07 ; +0.21] (n=50) | +0.27 [+0.16 ; +0.38] (n=49) | +0.17 [+0.08 ; +0.26] (n=46) | +0.28 [+0.17 ; +0.40] (n=45) | +0.33 [+0.22 ; +0.46] (n=43) | +0.35 [+0.22 ; +0.48] (n=36) |
| decay | +0.49 [+0.38 ; +0.60] (n=50) | +0.15 [+0.07 ; +0.24] (n=50) | +0.21 [+0.12 ; +0.31] (n=50) | +0.05 [+0.01 ; +0.10] (n=50) | +0.06 [+0.02 ; +0.12] (n=50) | +0.08 [+0.02 ; +0.16] (n=50) |

Dose brute moyenne appliquée par segment (avant `max_scale` 0,7 ; dose effective = dose brute x 0,7) :

| bras | seg 0 (0-31) | seg 1 (32-63) | seg 2 (64-95) | seg 3 (96-127) | seg 4 (128-159) | seg 5 (160-191) |
|---|---|---|---|---|---|---|
| neutral | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| constant | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pulse | 0.32 | 0.84 | 0.52 | 0.26 | 0.13 | 0.06 |
| constant_gen | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| decay | 0.73 | 0.36 | 0.18 | 0.09 | 0.05 | 0.02 |

## Sorties dégénérées

| bras | sorties | dégénérées | refus | < 192 tokens |
|---|---|---|---|---|
| neutral | 50 | 2 % | 2 % | 0 % |
| constant | 50 | 22 % | 0 % | 24 % |
| pulse | 50 | 0 % | 2 % | 0 % |
| constant_gen | 50 | 24 % | 2 % | 28 % |
| decay | 50 | 4 % | 0 % | 0 % |

## Critères de H6 (différences appariées, moyenne [IC95])

| critère | différence | IC disjoint de 0 (côté +) |
|---|---|---|
| pulse : ton du segment du pic - ton du dernier segment | -0.00 [-0.06 ; +0.05] (n=50) | non |
| pulse : ton du segment du pic - ton du premier segment | +0.02 [-0.04 ; +0.06] (n=50) | non |
| dernier segment : ton constant - ton pulse | +0.38 [+0.26 ; +0.52] (n=38) | oui |
| constant : ton du segment du pic - ton du dernier segment | +0.04 [-0.15 ; +0.23] (n=38) | non |
| premier segment : écart au neutre du constant - écart au neutre du pulse | +0.57 [+0.46 ; +0.67] (n=50) | oui |
| contrôle (a), premier segment : constant_gen - neutre | +0.19 [+0.11 ; +0.28] (n=50) | oui |
| contrôle (a), dernier segment : constant_gen - neutre | +0.35 [+0.23 ; +0.48] (n=36) | oui |
| premier segment : constant - constant_gen (part du prompt steeré) | +0.47 [+0.35 ; +0.58] (n=50) | oui |
| contrôle (b) : ton du premier segment - ton du dernier | +0.41 [+0.29 ; +0.53] (n=50) | oui |
| dernier segment : ton constant - ton decay | +0.34 [+0.21 ; +0.48] (n=38) | oui |

Lecture mécanique (montée : pic > premier segment ; redescente : pic > dernier segment ; début du pulse plus proche du neutre que celui du constant, IC95 apparié disjoint de 0 pour chacun) : **non**.

Contrôles : effet de la dose par token sans prompt steeré (constant_gen > neutre au premier ou au dernier segment) : **oui** ; décroissance seule lisible (decay : premier segment > dernier, et dernier segment sous le constant) : **oui**.

## Lecture

<!-- lecture -->
**Verdict H6 : non pour l'impulsion mesurée (montée 48, demi-vie 32) ; oui, sous une forme réduite, pour la décroissance seule.** Résultat central, pour la démo comme pour CAP-3 : le ton répond vite à la dose posée sur le prompt, et lentement à la dose posée pendant la génération. Les deux contrôles ajoutés à la revue (mêmes 50 prompts, même graine, GPU) le montrent.

- **Le pulse reste proche du neutre** (+0,02 à +0,07 selon les segments, contre -0,06 à -0,01 pour le neutre) : pic - dernier segment -0,00 [-0,06 ; +0,05], pic - premier segment +0,02 [-0,04 ; +0,06] (n=50). Ni montée ni redescente lisible.
- **Contrôle (a), constant_gen : la dose par token agit, mais elle s'installe lentement.** Avec le prompt évalué sans vecteur et la dose 1,0 dès le token 1, le ton n'est pas neutre : +0,13 au premier segment, puis +0,27, +0,17, +0,28, +0,33 et +0,35 au dernier (écart apparié au neutre +0,19 [+0,11 ; +0,28] au premier segment, +0,35 [+0,23 ; +0,48] au dernier, n=36). Il rejoint le bras constant en fin de sortie (+0,39), au bout d'une centaine de tokens. L'effet du constant ne vient donc pas seulement du prompt steeré.
- **Le prompt steeré fait le saut du début.** Au premier segment, le constant dépasse constant_gen de +0,47 [+0,35 ; +0,58] (n=50) : l'essentiel de l'effet immédiat du constant (+0,60) vient du vecteur posé sur les positions du prompt.
- **Contrôle (b), decay : la redescente se lit.** Pic 1,0 dès le prompt, puis décroissance seule de demi-vie 32 : le ton part de +0,49 [+0,38 ; +0,60], tombe à +0,15 puis +0,21, et finit entre +0,05 et +0,08 (premier - dernier segment +0,41 [+0,29 ; +0,53], n=50 ; au dernier segment, le constant reste au-dessus de +0,34 [+0,21 ; +0,48], n=38). Le ton suit la dose vers le bas, et finit un peu au-dessus du neutre. Le prompt, steeré à la dose 1, reste dans le cache pendant toute la sortie : sa trace ne suffit pas à tenir le ton.
- **Pourquoi le pulse ne se lit pas.** Le pulse laisse le prompt sans vecteur, comme constant_gen, et sa dose ne reste haute qu'une cinquantaine de tokens (moyenne 0,84 au segment 1, 0,26 au segment 3). D'après (a), c'est trop court pour que le ton s'installe : il faut une dose soutenue pendant une centaine de tokens. La décroissance, elle, a un effet rapide : le ton redescend dès le segment 1 du bras decay.
- **Ce que la démo peut montrer.** Une impulsion qui part du pic (montée 0) montre un ton joyeux au début qui se relâche ensuite. Une montée pendant la génération ne se lit pas sur 128 tokens avec ces réglages. Pour CAP-3, piloter le ton par un profil de dose pendant la génération demande des doses soutenues, ou une dose posée sur le prompt.
- **Sorties dégénérées.** Les deux bras à dose 1 soutenue dégénèrent dans environ un quart des cas (constant 22 %, constant_gen 24 %), contre 0 % pour le pulse et 4 % pour decay. Les sorties courtes (24 % et 28 %) n'alimentent pas les derniers segments, ce qui change la composition de ces segments (n=38 et n=36).
<!-- /lecture -->

## Limites

- Un seul modèle (Qwen3.5-4B quantifié Q4_K_M), un seul axe (joie ↔ tristesse), un seul profil
  d'impulsion (montée 48 tokens, demi-vie 32 tokens), une génération gloutonne par prompt.
- Le token 0 est tiré des logits du prompt : le contrôle constant_gen ne peut steerer qu'à partir
  du token 1, et le bras pulse (dose 0 au token 0) laisse lui aussi le prompt sans vecteur.
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
