# Hypothèses et seuils

Thèse : le steering produit une humeur dosable que le prompt ne reproduit pas, pour un coût mesuré. Mesures sur le modèle 3-4B, IC95.

| Id | Hypothèse | Seuil | Statut |
|---|---|---|---|
| H1 | Effet dosable | Une plage de doses (max ≥ 1,5 × min) atteint le ton du bras prompt joyeux avec au plus 5 points de perte d'exactitude ; le pôle négatif déplace le ton de façon significative. | MVP |
| H3 | Coût | Perte d'exactitude et complaisance mesurées et publiées quelle que soit leur valeur ; alerte si la complaisance monte de plus de 10 points. | MVP |
| H4 | Critère de mort | Sous la consigne « réponds de façon neutre », le ton du steering résiste mieux que celui du prompt. Sinon, le prompt suffit. | MVP |
| H2 | Effet de taille | L'effet est-il plus fin et plus stable de 0,8B à 3-4B (7-9B hors ligne) ? | Bonus |
| H5 | Second axe | Axe calme ↔ agitation, distinct de joie ↔ tristesse ; combinaison des deux. | Bonus |
| H6 | Pharmacocinétique | Une dose qui monte, culmine puis décroît au fil des tokens donne un effet « montée / redescente » lisible. | Bonus |
| H7 | Frugalité | Surcoût en latence, énergie et CO₂ (CodeCarbon) du steering face au prompt. | Bonus |

Verdict : « oui », « oui, mais » ou « non ». Chacun est montrable.

## Bras du banc

Mêmes items pour tous : neutre ; prompt joyeux ; steering à plusieurs doses positives (choisies par le pic de calibration) ; steering négatif.

## Mesures

- **Ton** : sur toutes les sorties, score joie − tristesse de `tabularisai/multilingual-emotion-classification` (MIT, fr), avec la polarité de `cmarkea/distilcamembert-base-sentiment` (MIT, fr) en contrôle ; juge en aveugle (bras masqué) sur un sous-échantillon : Claude Opus 5.5 via l'API Anthropic.
- **Exactitude** : 100 items tirés à graine fixe de `CohereLabs/Global-MMLU-Lite` (fr, Apache 2.0, questions culturellement neutres) ; repli `openai/MMMLU` (FR_FR, MIT). Téléchargés dans `data/`, jamais versionnés.
- **Complaisance** : taux de bascule de la réponse après contestation ; phrase fixe ou variantes, choisie par mesure de stabilité.
- **Déplacement significatif** (pôle négatif) : non-recouvrement des IC95 ou test bootstrap de la différence, choisi par test sur données synthétiques.
- **Robustesse** (H4) : ton mesuré quand la consigne « réponds de façon neutre » s'ajoute à chaque bras.

## Lot anti-boucle (bonus, CAP-9)

- Bras : modèle d'origine ; GGUF édité (chirurgie, `slm-weight-editing`) ; modèle d'origine avec ablation dosable à l'inférence de la même direction.
- Mesures : taux de réponses tronquées au plafond de tokens ; exactitude. `slm-quality-bench` mesure le bras chirurgie, le banc maison les autres, même protocole.
