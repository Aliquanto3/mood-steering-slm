# Rapport H2 : effet de taille

H2 (`hypotheses-et-seuils.md`) : l'effet est-il plus fin et plus stable de 0,8B à 3-4B ? Ce rapport relit les données de la campagne 3.5 (Qwen3.5-4B, 2B et 0.8B en Q4_K_M, aucune nouvelle génération). Il mesure l'expression d'humeur dans le texte produit, jamais un ressenti. Il reste descriptif : H2 est une hypothèse bonus, sans seuil de décision, et un écart que l'intervalle ne tranche pas est publié comme indéterminé.

**Le 7-9B n'a pas été mesuré**, faute de temps : la comparaison s'arrête à trois tailles de la même famille.

## Méthode

- Données : sorties de ton de la campagne 3.5 (`docs/rapport-verdict.md` et `docs/rapport-modeles-secondaires.md` donnent le protocole), 100 prompts ouverts en français, mêmes prompts et mêmes bras pour les trois modèles. Ton : score P(joie) − P(tristesse) du classifieur du banc. Le script lit `data/runs/campagne-3-5/` (hors dépôt) et ne la modifie pas.
- Doses : relatives au `max_scale` propre à chaque modèle (0,7 pour le 4B, 0,5 pour le 2B, 0,15 pour le 0.8B) ; les tableaux donnent la dose normalisée et l'échelle réelle (dose × `max_scale`). Les doses intermédiaires diffèrent : 0,43 et 0,71 (4B), 0,6 et 0,8 (2B), 0,67 et 0,83 (0.8B) ; seules les doses ±1 sont communes.
- **Dose utile** : un bras steering est utile si son ton est déplacé dans le sens de la dose (IC95 du bootstrap apparié de la différence de ton bras − neutre, sur les mêmes prompts, qui exclut 0 du côté de la dose ; même critère que les rapports du pulse et de l'axe calme) et si au plus 10 % de ses sorties sont dégénérées. Ce critère a remplacé, à la mise en commun v0.3, la disjonction de deux IC95 calculés séparément, plus conservatrice : sur ces données, aucun verdict n'a changé. Le seuil de 10 % est un choix de ce rapport, sans référence dans la SPEC ; sa sensibilité est publiée pour 5, 10 et 20 %. L'exactitude au QCM n'entre pas dans ce critère (H3 la traite). La plage utile est l'ensemble des doses utiles d'un pôle ; son « rapport » (plus forte dose utile / plus faible) dépend de la grille de doses, propre à chaque modèle, et ne se compare donc pas d'un modèle à l'autre.
- **Finesse** : un seul sens dans ce rapport, celui de la plage utile : un effet est plus fin s'il offre plus de doses utiles, c'est-à-dire plus de crans de ton distincts du neutre sans sorties dégénérées. La pente n'est pas une mesure de finesse.
- **Stabilité du ton** : écart-type du ton par bras (n − 1), IC95 de la moyenne, part de refus et de sorties dégénérées (`is_refusal` et `is_coherent` du pic 1.4, via `run_bench.output_flags`).
- **Pente dose → ton** : moindres carrés du ton moyen des bras d'un pôle, plus le neutre à la dose 0, calculée deux fois : sur la dose normalisée, puis sur l'échelle réelle (dose × `max_scale`). Elle mesure la sensibilité du ton à la dose, pas la finesse. Les deux versions sont publiées et comparées, car leur ordre entre modèles n'est pas le même.
- **Comparaison** : bootstrap percentile apparié à 2000 tirages. Les trois modèles ont reçu les mêmes 100 prompts (le script le vérifie et refuse une paire dont les prompts diffèrent) ; chaque tirage rééchantillonne les prompts une fois et calcule la statistique des deux modèles sur ces mêmes prompts, tous bras compris. Un écart est dit « non nul » si son IC95 exclut 0. Douze statistiques par paire, trois paires, sans correction de multiplicité : quelques écarts « non nuls » de borne proche de 0 sont à lire avec prudence.
- Reproduire : `uv run python scripts/report_size.py --out data/runs/campagne-3-5 --report docs/rapport-taille.md` (le bloc ci-dessous est généré ; le reste est écrit à la main).

<!-- mesures -->

## Mesures par modèle

### Qwen3.5-4B-Q4_K_M

`max_scale` 0.7, couches 8 à 24, 100 prompts de ton communs aux bras.

| Bras | Dose | Échelle réelle | Ton (IC95) | Écart-type | Refus | Dégénérées | Dose utile |
|---|---|---|---|---|---|---|---|
| neutral | +0 | +0 | 0.046 [0.017 ; 0.080] | 0.161 | 0 % | 2 % | — |
| prompt_joyful | +0 | +0 | 0.752 [0.685 ; 0.815] | 0.344 | 0 % | 0 % | — |
| steering_p0.43 | +0.4286 | +0.3 | 0.319 [0.251 ; 0.387] | 0.359 | 0 % | 0 % | oui |
| steering_p0.71 | +0.7143 | +0.5 | 0.464 [0.392 ; 0.538] | 0.372 | 0 % | 3 % | oui |
| steering_p1.00 | +1 | +0.7 | 0.702 [0.643 ; 0.759] | 0.305 | 0 % | 4 % | oui |
| steering_m0.43 | -0.4286 | -0.3 | -0.380 [-0.455 ; -0.304] | 0.393 | 0 % | 1 % | oui |
| steering_m0.71 | -0.7143 | -0.5 | -0.544 [-0.617 ; -0.469] | 0.399 | 5 % | 19 % | non |
| steering_m1.00 | -1 | -0.7 | -0.599 [-0.681 ; -0.518] | 0.409 | 5 % | 49 % | non |

Plage de doses utile :

- pôle positif : doses 0.4286, 0.7143, 1 (rapport 2.33) ; échelles 0.3, 0.5, 0.7
- pôle négatif : doses 0.4286 ; échelles 0.3

Sensibilité au seuil de sorties dégénérées (doses utiles par pôle) :

- seuil 5 % : positif 0.4286, 0.7143, 1 ; négatif 0.4286
- seuil 10 % : positif 0.4286, 0.7143, 1 ; négatif 0.4286
- seuil 20 % : positif 0.4286, 0.7143, 1 ; négatif 0.4286, 0.7143

Pente dose → ton (tous les bras d'un pôle, plus le neutre à la dose 0) :

- pôle positif : 0.641 [0.579 ; 0.705] par dose normalisée, 0.915 [0.827 ; 1.007] par unité d'échelle réelle
- pôle négatif : 0.658 [0.574 ; 0.744] par dose normalisée, 0.940 [0.819 ; 1.063] par unité d'échelle réelle
- tous bras : 0.685 [0.632 ; 0.736] par dose normalisée, 0.978 [0.903 ; 1.051] par unité d'échelle réelle

Face au prompt : ton du prompt joyeux 0.752 [0.685 ; 0.815], ton à la dose +1 0.702 [0.643 ; 0.759] : l'IC95 du steering recouvre celui du prompt.

### Qwen3.5-2B-Q4_K_M

`max_scale` 0.5, couches 6 à 18, 100 prompts de ton communs aux bras.

| Bras | Dose | Échelle réelle | Ton (IC95) | Écart-type | Refus | Dégénérées | Dose utile |
|---|---|---|---|---|---|---|---|
| neutral | +0 | +0 | 0.015 [-0.017 ; 0.047] | 0.166 | 0 % | 5 % | — |
| prompt_joyful | +0 | +0 | 0.505 [0.434 ; 0.577] | 0.371 | 10 % | 2 % | — |
| steering_p0.60 | +0.6 | +0.3 | 0.238 [0.176 ; 0.302] | 0.325 | 0 % | 3 % | oui |
| steering_p0.80 | +0.8 | +0.4 | 0.324 [0.257 ; 0.395] | 0.360 | 0 % | 11 % | non |
| steering_p1.00 | +1 | +0.5 | 0.353 [0.279 ; 0.429] | 0.385 | 0 % | 33 % | non |
| steering_m0.60 | -0.6 | -0.3 | -0.445 [-0.519 ; -0.371] | 0.390 | 8 % | 17 % | non |
| steering_m0.80 | -0.8 | -0.4 | -0.409 [-0.482 ; -0.334] | 0.381 | 17 % | 36 % | non |
| steering_m1.00 | -1 | -0.5 | -0.435 [-0.510 ; -0.359] | 0.390 | 13 % | 61 % | non |

Plage de doses utile :

- pôle positif : doses 0.6 ; échelles 0.3
- pôle négatif : aucune

Sensibilité au seuil de sorties dégénérées (doses utiles par pôle) :

- seuil 5 % : positif 0.6 ; négatif aucune
- seuil 10 % : positif 0.6 ; négatif aucune
- seuil 20 % : positif 0.6, 0.8 ; négatif 0.6

Pente dose → ton (tous les bras d'un pôle, plus le neutre à la dose 0) :

- pôle positif : 0.353 [0.281 ; 0.423] par dose normalisée, 0.705 [0.561 ; 0.845] par unité d'échelle réelle
- pôle négatif : 0.472 [0.392 ; 0.549] par dose normalisée, 0.945 [0.783 ; 1.097] par unité d'échelle réelle
- tous bras : 0.446 [0.398 ; 0.494] par dose normalisée, 0.892 [0.797 ; 0.987] par unité d'échelle réelle

Face au prompt : ton du prompt joyeux 0.505 [0.434 ; 0.577], ton à la dose +1 0.353 [0.279 ; 0.429] : l'IC95 du steering reste sous celui du prompt.

### Qwen3.5-0.8B-Q4_K_M

`max_scale` 0.15, couches 6 à 18, 100 prompts de ton communs aux bras.

| Bras | Dose | Échelle réelle | Ton (IC95) | Écart-type | Refus | Dégénérées | Dose utile |
|---|---|---|---|---|---|---|---|
| neutral | +0 | +0 | -0.009 [-0.046 ; 0.028] | 0.190 | 1 % | 11 % | — |
| prompt_joyful | +0 | +0 | 0.465 [0.396 ; 0.543] | 0.380 | 0 % | 23 % | — |
| steering_p0.67 | +0.6667 | +0.1 | 0.223 [0.165 ; 0.285] | 0.305 | 0 % | 19 % | non |
| steering_p0.83 | +0.8333 | +0.125 | 0.336 [0.267 ; 0.408] | 0.367 | 0 % | 25 % | non |
| steering_p1.00 | +1 | +0.15 | 0.305 [0.238 ; 0.374] | 0.356 | 0 % | 36 % | non |
| steering_m0.67 | -0.6667 | -0.1 | -0.242 [-0.307 ; -0.180] | 0.334 | 8 % | 52 % | non |
| steering_m0.83 | -0.8333 | -0.125 | -0.261 [-0.325 ; -0.200] | 0.321 | 6 % | 70 % | non |
| steering_m1.00 | -1 | -0.15 | -0.300 [-0.367 ; -0.235] | 0.336 | 5 % | 82 % | non |

Plage de doses utile :

- pôle positif : aucune
- pôle négatif : aucune

Sensibilité au seuil de sorties dégénérées (doses utiles par pôle) :

- seuil 5 % : positif aucune ; négatif aucune
- seuil 10 % : positif aucune ; négatif aucune
- seuil 20 % : positif 0.6667 ; négatif aucune

Pente dose → ton (tous les bras d'un pôle, plus le neutre à la dose 0) :

- pôle positif : 0.346 [0.275 ; 0.420] par dose normalisée, 2.304 [1.830 ; 2.799] par unité d'échelle réelle
- pôle négatif : 0.298 [0.231 ; 0.362] par dose normalisée, 1.984 [1.542 ; 2.411] par unité d'échelle réelle
- tous bras : 0.330 [0.286 ; 0.376] par dose normalisée, 2.201 [1.907 ; 2.504] par unité d'échelle réelle

Face au prompt : ton du prompt joyeux 0.465 [0.396 ; 0.543], ton à la dose +1 0.305 [0.238 ; 0.374] : l'IC95 du steering reste sous celui du prompt.

## Comparaisons entre modèles

### Qwen3.5-4B-Q4_K_M moins Qwen3.5-2B-Q4_K_M

| Statistique | Écart (IC95 bootstrap) | Lecture |
|---|---|---|
| pente dose → ton, pôle positif (par dose normalisée) | 0.288 [0.213 ; 0.367] | écart non nul |
| pente dose → ton, pôle négatif (par dose normalisée) | 0.186 [0.095 ; 0.271] | écart non nul |
| pente dose → ton, tous bras (par dose normalisée) | 0.239 [0.191 ; 0.289] | écart non nul |
| pente dose → ton, pôle positif (par unité d'échelle réelle) | 0.210 [0.070 ; 0.357] | écart non nul |
| pente dose → ton, pôle négatif (par unité d'échelle réelle) | -0.005 [-0.168 ; 0.144] | indéterminé (l'IC contient 0) |
| pente dose → ton, tous bras (par unité d'échelle réelle) | 0.086 [0.004 ; 0.171] | écart non nul |
| ton à la dose +1 | 0.348 [0.270 ; 0.428] | écart non nul |
| ton à la dose −1 | -0.164 [-0.250 ; -0.080] | écart non nul |
| écart-type du ton à la dose +1 | -0.080 [-0.132 ; -0.023] | écart non nul |
| écart-type du ton à la dose −1 | 0.020 [-0.020 ; 0.055] | indéterminé (l'IC contient 0) |
| part de sorties dégénérées à +1 | -0.290 [-0.390 ; -0.200] | écart non nul |
| part de sorties dégénérées à −1 | -0.120 [-0.230 ; -0.010] | écart non nul |

### Qwen3.5-4B-Q4_K_M moins Qwen3.5-0.8B-Q4_K_M

| Statistique | Écart (IC95 bootstrap) | Lecture |
|---|---|---|
| pente dose → ton, pôle positif (par dose normalisée) | 0.295 [0.209 ; 0.385] | écart non nul |
| pente dose → ton, pôle négatif (par dose normalisée) | 0.360 [0.254 ; 0.467] | écart non nul |
| pente dose → ton, tous bras (par dose normalisée) | 0.355 [0.302 ; 0.410] | écart non nul |
| pente dose → ton, pôle positif (par unité d'échelle réelle) | -1.388 [-1.874 ; -0.923] | écart non nul |
| pente dose → ton, pôle négatif (par unité d'échelle réelle) | -1.044 [-1.502 ; -0.585] | écart non nul |
| pente dose → ton, tous bras (par unité d'échelle réelle) | -1.223 [-1.499 ; -0.950] | écart non nul |
| ton à la dose +1 | 0.396 [0.326 ; 0.471] | écart non nul |
| ton à la dose −1 | -0.299 [-0.396 ; -0.199] | écart non nul |
| écart-type du ton à la dose +1 | -0.051 [-0.107 ; 0.007] | indéterminé (l'IC contient 0) |
| écart-type du ton à la dose −1 | 0.074 [0.024 ; 0.121] | écart non nul |
| part de sorties dégénérées à +1 | -0.320 [-0.420 ; -0.220] | écart non nul |
| part de sorties dégénérées à −1 | -0.330 [-0.450 ; -0.210] | écart non nul |

### Qwen3.5-2B-Q4_K_M moins Qwen3.5-0.8B-Q4_K_M

| Statistique | Écart (IC95 bootstrap) | Lecture |
|---|---|---|
| pente dose → ton, pôle positif (par dose normalisée) | 0.007 [-0.084 ; 0.092] | indéterminé (l'IC contient 0) |
| pente dose → ton, pôle négatif (par dose normalisée) | 0.175 [0.078 ; 0.274] | écart non nul |
| pente dose → ton, tous bras (par dose normalisée) | 0.116 [0.071 ; 0.161] | écart non nul |
| pente dose → ton, pôle positif (par unité d'échelle réelle) | -1.598 [-2.069 ; -1.141] | écart non nul |
| pente dose → ton, pôle négatif (par unité d'échelle réelle) | -1.039 [-1.479 ; -0.583] | écart non nul |
| pente dose → ton, tous bras (par unité d'échelle réelle) | -1.309 [-1.574 ; -1.046] | écart non nul |
| ton à la dose +1 | 0.048 [-0.037 ; 0.137] | indéterminé (l'IC contient 0) |
| ton à la dose −1 | -0.135 [-0.225 ; -0.037] | écart non nul |
| écart-type du ton à la dose +1 | 0.029 [-0.018 ; 0.076] | indéterminé (l'IC contient 0) |
| écart-type du ton à la dose −1 | 0.054 [0.014 ; 0.098] | écart non nul |
| part de sorties dégénérées à +1 | -0.030 [-0.150 ; 0.090] | indéterminé (l'IC contient 0) |
| part de sorties dégénérées à −1 | -0.210 [-0.320 ; -0.100] | écart non nul |

<!-- /mesures -->

## Lecture

- **Plage de doses utile (finesse).** Sur le 4B, les trois doses positives (échelles 0,3, 0,5 et 0,7) sont utiles ; au pôle négatif, seule la dose −0,43 l'est, car les sorties dégénérées passent à 19 % puis 49 %. Sur le 2B, seule la dose +0,6 (échelle 0,3) l'est : +0,8 dépasse le seuil de peu, à 11 %, +1 monte à 33 %, et aucune dose négative ne passe (17 % de dégénérées dès −0,6). Sur le 0.8B, aucune dose ne l'est, alors que le ton se déplace de façon nette à toutes les doses : le critère échoue sur les sorties dégénérées (19 % dès +0,67, 52 % dès −0,67), le bras neutre en comptant déjà 11 %. Le classement dépend du seuil : à 5 % comme à 10 %, le 4B a 3 + 1 doses utiles, le 2B 1 + 0 et le 0.8B aucune ; à 20 %, le 4B en a 3 + 2, le 2B 2 + 1 et le 0.8B 1 + 0 (+0,67). L'ordre 4B > 2B > 0.8B tient donc aux trois seuils, l'écart entre 2B et 0.8B tient à 5 et 10 % (la dose +0,6 du 2B n'a que 3 % de dégénérées, la meilleure du 0.8B 19 %) et se resserre à 20 %. Le nombre de doses utiles, plus que leur rapport, se compare entre modèles : les grilles de doses diffèrent.
- **Sensibilité (pente).** À l'échelle normalisée, la pente du 4B vaut 0,64 (pôle positif) et 0,66 (négatif), contre 0,35 et 0,47 pour le 2B et 0,35 et 0,30 pour le 0.8B ; le 4B dépasse le 2B de 0,29 [0,21 ; 0,37] et 0,19 [0,10 ; 0,27], et le 0.8B de 0,30 [0,21 ; 0,39] et 0,36 [0,25 ; 0,47]. Par unité d'échelle réelle, la pente vaut 0,92 et 0,94 (4B), 0,71 et 0,95 (2B), 2,30 et 1,98 (0.8B) : le 0.8B passe devant les deux autres (4B − 0.8B −1,39 [−1,87 ; −0,92] et −1,04 [−1,50 ; −0,59] ; 2B − 0.8B −1,60 [−2,07 ; −1,14] et −1,04 [−1,48 ; −0,58]), et entre 4B et 2B l'écart n'est non nul qu'au pôle positif (+0,21 [0,07 ; 0,36]), indéterminé au négatif (−0,005 [−0,17 ; 0,14]). Le sens de la comparaison dépend donc de la normalisation : la dose normalisée suit le `max_scale` choisi pour chaque modèle au pic 1.4 (0,7, 0,5 et 0,15), l'échelle réelle ignore que ce plafond a été fixé là où les sorties restaient lisibles. Aucune des deux lectures ne sépare la taille de ce réglage ; la pente n'entre pas dans la conclusion sur la finesse.
- **Amplitude à ±1.** Le ton à la dose +1 vaut 0,70 (4B), 0,35 (2B) et 0,31 (0.8B) : le 4B est au-dessus des deux autres (+0,35 [0,27 ; 0,43] et +0,40 [0,33 ; 0,47]), le 2B et le 0.8B ne se distinguent pas (+0,05 [−0,04 ; 0,14]). À −1, le ton vaut −0,60, −0,44 et −0,30 (écarts 4B−2B −0,16 [−0,25 ; −0,08], 2B−0.8B −0,14 [−0,23 ; −0,04]), mais avec 49 %, 61 % et 82 % de sorties dégénérées : le ton d'un texte dégénéré est mal défini, ces écarts ne mesurent pas un contrôle plus fin.
- **Stabilité.** À +1, la part de sorties dégénérées vaut 4 %, 33 % et 36 % : le 4B est nettement plus stable que les deux autres (−0,29 [−0,39 ; −0,20] face au 2B, −0,32 [−0,42 ; −0,22] face au 0.8B), le 2B et le 0.8B sont indéterminés. À −1, elle vaut 49 %, 61 % et 82 % : écart non nul entre 4B et 0.8B (−0,33 [−0,45 ; −0,21]) et entre 2B et 0.8B (−0,21 [−0,32 ; −0,10]), borne proche de 0 entre 4B et 2B (−0,12 [−0,23 ; −0,01]). L'écart-type du ton à ±1 ne donne pas de lecture nette : plus faible pour le 4B à +1 (0,31 contre 0,39 pour le 2B, écart −0,08 [−0,13 ; −0,02]), indéterminé entre 4B et 2B à −1, et un écart non nul entre 4B et 0.8B seulement à −1 (+0,07 [0,02 ; 0,12]) dans le sens du 4B plus dispersé.
- **Ce que le test tranche.** Le 4B a moins de sorties dégénérées à +1 que le 2B et le 0.8B (IC disjoints de 0). En décompte descriptif, sans IC sur l'écart, il a aussi davantage de doses utiles à chacun des trois seuils ; son ton à +1 est aussi plus marqué. Entre le 2B et le 0.8B, aucun écart établi par le bootstrap au pôle positif, et le 2B ne garde qu'une ou deux doses utiles de plus selon le seuil, proches de ce seuil ; le 0.8B est plus dégradé au pôle négatif. La pente ne tranche rien : son ordre s'inverse selon la normalisation. Aucune tendance monotone en taille n'est testée : trois points ne le permettent pas. Les données sont compatibles avec « plus fin (plus de doses utiles) et plus stable à 3-4B », sans établir un effet de taille.

## Limites

- **`max_scale` différent par modèle** (0,7, 0,5, 0,15), fixé au pic 1.4 pour chacun : la dose normalisée 1 ne désigne pas le même effort de steering, et la taille se confond avec ce réglage et avec les couches ciblées (8 à 24 pour le 4B, 6 à 18 pour les deux autres).
- **Trois tailles seulement**, 0,8B, 2B et 4B, et une seule famille (Qwen3.5, Q4_K_M) : aucune extrapolation à d'autres familles ni à des modèles plus grands.
- **Aucun 7-9B**, faute de temps ; le bas de l'hypothèse (« de 0,8B à 3-4B ») est couvert, pas le haut.
- Un seul classifieur de ton, sans juge en aveugle pour les modèles secondaires ; le ton d'un texte dégénéré est peu fiable.
- 100 prompts issus de gabarits × sujets, donc non indépendants ; une seule génération gloutonne par prompt. Le seuil de 10 % de dégénérées (sensibilité publiée à 5 et 20 %) et le critère de dose utile sont des choix de ce rapport. Pas de correction de multiplicité sur les 36 comparaisons.
- Les doses intermédiaires diffèrent d'un modèle à l'autre ; seules les doses ±1 se comparent point à point.
