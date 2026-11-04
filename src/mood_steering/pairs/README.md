# Paires contrastées

Textes d'auteur rédigés à la main pour le projet, versionnés (aucune donnée tierce).

`joy_sadness.jsonl` : axe joie ↔ tristesse, 100 paires en français.

Format : une ligne JSON par paire, deux clés seules, `positive` (pôle joie) et `negative` (pôle tristesse), chaînes non vides sur une ligne. Aucun texte n'apparaît deux fois dans le fichier. Les deux phrases d'une paire gardent autant que possible la même structure : seule l'humeur exprimée change, pour que la différence d'activations porte sur le ton et non sur le sujet.

```json
{"positive": "Quelle belle journée, le soleil brille et je souris.", "negative": "Quelle journée grise, la pluie tombe et je soupire."}
```

`llama-cvector-generator` les reçoit en deux fichiers d'une phrase par ligne (`--positive-file`, `--negative-file`), dans le même ordre ; `spikes/calibration_scan.py vector` fait la conversion. Le vecteur obtenu sert à faire exprimer une humeur au modèle, jamais à lui prêter un ressenti.
