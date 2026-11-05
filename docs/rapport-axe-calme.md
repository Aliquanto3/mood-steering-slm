# Rapport H5 : second axe calme ↔ agitation

> **Verdict H5 : non tenue** (Qwen3.5-4B-Q4_K_M, un seul modèle). Expression d'une humeur dans le texte produit, jamais un ressenti du modèle. Paires non encore relues.

## Méthode

- Modèle : `Qwen3.5-4B-Q4_K_M`, couches 8 à 24, `max_scale` 0.7. Vecteurs : `llama-cvector-generator --method mean`, `src/mood_steering/pairs/joy_sadness.jsonl` (100 paires) et `src/mood_steering/pairs/calm_agitation.jsonl` (paires calme posé contre agité ou pressé, valence proche ; **non relues par Anaël**).
- llama.cpp n'accepte qu'un vecteur de contrôle par contexte (le suivant écrase le précédent) : le bras combiné pose `s_joie × v_joie + s_calme × v_calme`, somme faite côté numpy.
- Bras (mêmes 50 prompts neutres du banc, 128 tokens, génération gloutonne, graine `seed + index` enregistrée sans effet) : neutre ; par échelle s ∈ {0.2, 0.4} (dose × `max_scale`) : calme+ (calme = +s), agitation (calme = −s), joie+ seule (joie = +s), combiné (joie = +s et calme = +s).
- Génération : llama-cpp-python 0.3.36, CPU. Aucun juge, aucun appel payant.
- **Proxies d'activation (aucun classifieur calme/agitation n'existe)** : (1) `act_cls` = part de P(colère)+P(peur)+P(surprise) dans la somme des probabilités de `tabularisai/multilingual-emotion-classification` ; (2) `exclam` = « ! » par phrase ; (3) `lex` = (mots d'agitation − mots de calme) pour 100 mots, lexique de préfixes dans `scripts/run_axis.py` ; (4) `sent_len` = longueur moyenne des phrases, descriptive seulement (sens non établi). **Indice d'activation** = moyenne des écarts au neutre (apparié par prompt) de (1) à (3), chacun divisé par son écart-type sur le bras neutre : plus = plus agité.
- Valence = P(joie) − P(tristesse) du même classifieur, écart au neutre divisé par l'écart-type neutre. Activation et valence sont donc en unités d'écart-type du neutre, comparables entre elles.
- IC95 : bootstrap percentile apparié par prompt (2000 tirages, graine 0), prompts rééchantillonnés ensemble pour tous les indicateurs d'un bras.

### Critères fixés avant la mesure

- **Distinction** : |cos| < 0.7 sur chacune des couches 8 à 24 (cos² < 0.49 : moins de la moitié du carré de la norme d'un axe est portée par l'autre ; au-dessus, les deux axes seraient pour l'essentiel colinéaires). Deux directions aléatoires en dimension 2560 ont |cos| ≈ 0,02 : le seuil tolère le recouvrement attendu de deux axes d'émotion sans les confondre.
- **H5 (a)** : à une échelle donnée, (calme+) l'indice d'activation baisse, IC95 < 0, et |Δactivation| − |Δvalence| > 0 (IC95 > 0) ; (agitation) l'activation monte, IC95 > 0, même sélectivité ; (joie+) la valence monte, IC95 > 0, et |Δvaleur| > |Δactivation| (IC95 de |Δact| − |Δval| < 0).
- **H5 (b)** : le bras combiné garde l'activation de calme+ (IC95 < 0) et la valence de joie+ (IC95 > 0), chaque effet à au moins 50% de l'effet du bras seul (rapport des moyennes).
- **Validité** d'une échelle : chaque bras dépasse le neutre de 20% de sorties dégénérées au plus (`is_coherent` du pic 1.4).
- **Verdict** : tenue si les cosinus passent et si (a), (b) et la validité sont réunis à au moins une échelle ; partielle si une partie seulement l'est ; sinon non tenue.

## Cosinus entre calm_agitation et joy_sadness

| Couche | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cos | +0.30 | +0.30 | +0.29 | +0.28 | +0.28 | +0.30 | +0.29 | +0.30 | +0.29 | +0.29 | +0.29 | +0.33 | +0.32 | +0.31 | +0.36 | +0.28 | +0.36 |

|cos| maximal 0.36, médian 0.30 (seuil 0.7) : l'axe **se distingue** de joie ↔ tristesse par ce critère.

## Mesures par bras

Écarts au neutre, moyenne [IC95], en écarts-types du neutre (n = 50 par bras). Écart-type neutre : act_cls 0.117, exclam 0.135, lex 0.979, valence 0.238. Dégénérées du neutre : 2 %.

| Bras | Activation | Valence | |Δact|−|Δval| | Dégénérées | Refus |
|---|---|---|---|---|---|
| calm_p0.20 | -0.50 [-0.70 ; -0.32] | -0.29 [-0.55 ; -0.06] | +0.21 [-0.09 ; +0.52] | 2 % | 0 % |
| agit_m0.20 | +0.48 [+0.32 ; +0.65] | -0.17 [-0.48 ; +0.12] | +0.32 [+0.01 ; +0.52] | 0 % | 0 % |
| joy_p0.20 | +0.63 [+0.45 ; +0.81] | +0.82 [+0.31 ; +1.31] | -0.19 [-0.71 ; +0.34] | 0 % | 0 % |
| joycalm_p0.20 | +0.25 [+0.08 ; +0.42] | +0.75 [+0.28 ; +1.18] | -0.49 [-0.93 ; -0.02] | 0 % | 0 % |
| calm_p0.40 | -0.90 [-1.19 ; -0.63] | -0.22 [-0.56 ; +0.07] | +0.68 [+0.26 ; +1.06] | 56 % | 0 % |
| agit_m0.40 | +2.49 [+0.93 ; +5.40] | -0.14 [-0.41 ; +0.12] | +2.35 [+0.67 ; +5.30] | 14 % | 0 % |
| joy_p0.40 | +1.29 [+1.08 ; +1.49] | +1.42 [+1.01 ; +1.84] | -0.13 [-0.56 ; +0.30] | 0 % | 0 % |
| joycalm_p0.40 | -0.10 [-0.30 ; +0.10] | +0.01 [-0.39 ; +0.40] | +0.09 [-0.36 ; +0.22] | 94 % | 0 % |

### Proxies d'activation, écarts bruts au neutre

| Bras | act_cls | « ! » par phrase | lexique (pour 100 mots) | longueur de phrase (mots) | valence brute |
|---|---|---|---|---|---|
| calm_p0.20 | -0.013 [-0.052 ; +0.013] | -0.049 [-0.074 ; -0.026] | -1.00 [-1.55 ; -0.52] | +0.96 [-0.11 ; +2.03] | -0.068 [-0.130 ; -0.015] |
| agit_m0.20 | -0.009 [-0.047 ; +0.013] | +0.041 [+0.011 ; +0.076] | +1.20 [+0.73 ; +1.65] | -0.04 [-1.21 ; +1.04] | -0.040 [-0.113 ; +0.029] |
| joy_p0.20 | -0.017 [-0.051 ; +0.001] | +0.230 [+0.178 ; +0.282] | +0.33 [+0.04 ; +0.62] | -2.26 [-3.34 ; -1.16] | +0.195 [+0.073 ; +0.311] |
| joycalm_p0.20 | -0.018 [-0.053 ; +0.000] | +0.190 [+0.130 ; +0.250] | -0.48 [-0.86 ; -0.13] | -1.64 [-3.03 ; +0.04] | +0.178 [+0.066 ; +0.281] |
| calm_p0.40 | +0.003 [-0.035 ; +0.028] | -0.075 [-0.112 ; -0.038] | -2.12 [-2.91 ; -1.41] | +3.94 [-0.61 ; +9.03] | -0.052 [-0.133 ; +0.016] |
| agit_m0.40 | +0.074 [+0.020 ; +0.132] | +0.667 [+0.073 ; +1.831] | +1.85 [+1.34 ; +2.35] | -0.56 [-2.02 ; +0.88] | -0.033 [-0.097 ; +0.028] |
| joy_p0.40 | -0.018 [-0.053 ; -0.000] | +0.494 [+0.439 ; +0.549] | +0.36 [-0.01 ; +0.71] | -3.71 [-4.76 ; -2.68] | +0.339 [+0.241 ; +0.438] |
| joycalm_p0.40 | +0.057 [+0.005 ; +0.099] | -0.086 [-0.130 ; -0.036] | -0.16 [-0.44 ; +0.10] | +39.69 [+30.47 ; +50.07] | +0.003 [-0.093 ; +0.095] |

## Critère H5 par échelle

| Échelle | Valide | calme+ | agitation | joie+ | (a) | combiné : activation | combiné : valence | rétention act. | rétention val. | (b) | Tenue |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 | oui | non | oui | non | non | non | oui | -0.51 | 0.91 | non | non |
| 0.4 | non | oui | oui | non | non | non | non | 0.12 | 0.01 | non | non |

## Extraits (premier prompt, 140 caractères)

- neutral : « Le marché du samedi matin est un vibrant carrefour où l'odeur de la cuisine traditionnelle se mêle au bruit animé des négociations. Des cent… »
- calm_p0.20 : « Le marché du samedi matin est un lieu animé où les bruits des vendeurs et des acheteurs se mêlent à l'odeur des produits frais. Les étals so… »
- agit_m0.20 : « Le marché du samedi matin est un spectacle coloré où les vendeurs déploient leurs étals bondés de fruits, de légumes frais et de produits ar… »
- joy_p0.20 : « Le marché du samedi matin est un vibrant carrefour où les couleurs des étals s'animent sous le soleil, attirant une foule diverse pour ses p… »
- joycalm_p0.20 : « Le marché du samedi matin est une scène animée où les bruits de rires, les odeurs de produits frais et les couleurs vives créent une ambianc… »
- calm_p0.40 : « Le marché du samedi matin est un lieu paisible et animé où les vendeurs disposent leurs produits frais, souvent avec des fleurs, des fruits,… »
- agit_m0.40 : « Le samedi matin, les trottoers se transforment en un immense carrefour où les vendeurs hurlent leurs prix pour des fruits, légumes et friand… »
- joy_p0.40 : « Le marché du samedi matin est une véritable célébration de la vie locale où les couleurs et les odeurs s'entremêlent pour créer une ambiance… »
- joycalm_p0.40 : « 这是一个 33<think> <think> 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 33 »

## Lecture

<!-- lecture -->
**Distinction : oui.** Le cosinus entre `calm_agitation` et `joy_sadness` vaut de +0,28 à +0,36 sur les couches 8 à 24 (seuil 0,7) : les directions sont distinctes, avec un recouvrement positif modéré et stable d'une couche à l'autre.

**Écart de protocole (à lire d'abord).** Les échelles prévues (0,3 à 0,5) n'ont pas été mesurées. Le GPU était occupé par d'autres campagnes (moins de 3 Gio libres, puis plus du tout) : la génération a tourné sur CPU, par tranches reprises, avec 50 prompts et 128 tokens comme prévu mais deux échelles seulement, **0,2 et 0,4**. À 0,4, le vecteur de calme seul rend 56 % de sorties dégénérées et le bras combiné 94 % (échelle totale du vecteur sommé ≈ 0,65, au-delà de ce que le 4B tolère) : l'échelle 0,4 est invalide pour (a) et (b). L'échelle 0,2 a donc été ajoutée : c'est la seule valide (0 à 2 % de dégénérées). La plage utile du calme est plus étroite que celle de la joie.

**(a) non tenue à 0,2.** Calme+ baisse l'activation (−0,50 écart-type [−0,70 ; −0,32]) et l'agitation la monte (+0,48 [+0,32 ; +0,65]), mais le critère de sélectivité échoue : pour calme+, |Δact| − |Δval| = +0,21 [−0,09 ; +0,52], l'IC contient zéro, car calme+ baisse aussi un peu la valence (−0,29 [−0,55 ; −0,06]) ; pour l'agitation, la sélectivité passe de justesse (+0,32 [+0,01 ; +0,52]). Surtout, **joie+ ne déplace pas la valence plus que l'activation** (valence +0,82, activation +0,63 ; |Δact| − |Δval| = −0,19 [−0,71 ; +0,34]) : l'indice d'activation monte avec la joie, surtout par les « ! » (+0,23 par phrase), qui sont autant un marqueur d'enthousiasme que d'agitation. C'est la limite du proxy plus qu'une preuve contre l'axe.

**(b) non tenue à 0,2.** Le bras combiné garde la valence de joie+ (+0,75 [+0,28 ; +1,18], rétention 0,91) mais pas l'activation de calme+ : elle reste positive (+0,25 [+0,08 ; +0,42], rétention −0,51), les « ! » apportés par la joie compensant le calme. Le lexique, lui, baisse (−0,48) : les proxies divergent.

**Ce que le résultat permet de dire.** Les deux directions sont distinctes et agissent dans le sens attendu sur le lexique et les « ! » (calme+ réduit les deux, l'agitation les augmente). Le critère H5 tel que fixé n'est pas satisfait : l'activation mesurée par ces proxies n'est pas indépendante de la joie, et la combinaison ne conserve pas l'effet de calme. Une lecture plus fine demande un meilleur proxy d'activation (classifieur dédié ou juge), un seul modèle et un seul tirage glouton ne suffisant pas à conclure au-delà de ce banc. Les échelles 0,3 et 0,5 restent à mesurer sur GPU.
<!-- /lecture -->

## Limites

- **Proxies d'activation** : aucun classifieur calme/agitation n'a été validé ; `act_cls` vient du même classifieur que la valence (les deux sont corrélés par construction), `exclam` et le lexique sont grossiers. Une activation « mesurée » est un indice de style du texte, pas un état.
- Un seul modèle (Qwen3.5-4B Q4_K_M), un seul jeu de 50 prompts neutres, génération gloutonne : un seul tirage par prompt.
- Les paires `calm_agitation.jsonl` n'ont pas été relues par Anaël ; les contrastes (calme posé contre agité ou pressé) gardent une valence proche, sans garantie.
- Les cosinus portent sur des différences de moyennes (méthode `mean`) à 107 paires ; ils mesurent la proximité des directions, pas l'indépendance des effets sur le texte.
