# Démo : lancement et scénario

La démo met trois colonnes côte à côte pour un même prompt : le modèle seul (**Neutre**), le modèle avec une consigne de jeu de rôle joyeux (**Prompt joyeux**) et le modèle dont les activations sont poussées par un vecteur (**Steering d'activations**), à une dose réglable de −1 à +1. Elle montre l'expression d'une humeur dans le texte produit, jamais un ressenti du modèle.

Verdict de la campagne 3.5 sur le Qwen3.5-4B, en deux lignes : le steering n'égale le prompt joyeux qu'à la dose +1 (verdict « non » pour H1), mais il résiste à une consigne neutre là où le prompt cède (H4 : oui). Le pôle négatif produit des sorties dégénérées à forte dose ; détail dans [`rapport-verdict.md`](rapport-verdict.md).

## Prérequis

- Python 3.11 ou plus et [uv](https://docs.astral.sh/uv/).
- Les dépendances de la démo avec le moteur llama.cpp : `uv sync --extra llama` (roue CPU de `llama-cpp-python`, aucune compilation).
- Le modèle : `Qwen3.5-4B-Q4_K_M.gguf` (dépôt Hugging Face `unsloth/Qwen3.5-4B-GGUF`, 2,74 Go sur disque, soit 2,55 Gio) dans `models/gguf/`. Le dossier est ignoré par git.
- Le vecteur de steering : `data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf` (ignoré par git, 319 Ko mesurés). Pour le produire :
  1. obtenir `llama-cvector-generator` de llama.cpp (build b12118, celui de la campagne : binaire à télécharger dans les releases de llama.cpp ou à compiler) et indiquer son chemin complet dans la variable `MOOD_CVECTOR_GENERATOR`, dans le même terminal que la commande suivante :
     - PowerShell : `$env:MOOD_CVECTOR_GENERATOR = "C:\chemin\llama-cvector-generator.exe"`
     - bash : `export MOOD_CVECTOR_GENERATOR=/chemin/llama-cvector-generator`
  2. lancer `uv run python scripts/run_bench.py --regenerate-vectors --models 4b` (100 paires de `src/mood_steering/pairs/joy_sadness.jsonl`, méthode `mean`, 4 à 7 minutes).
- Mémoire : le fichier du modèle pèse 2,74 Go sur disque ; la campagne a mesuré 3 387 Mio de VRAM pour le 4B sur GPU à 4096 de contexte (`pics/choix-des-modeles.md`), soit un ordre de grandeur de 3,4 Gio en mémoire, hors cache supplémentaire d'une instance par colonne. Non mesuré ici pour la démo sur CPU : prévoir une marge sur la RAM.

Sans vecteur, la démo reste utilisable : voir « Repli sans vecteur ».

## Lancement

```
uv run python -m mood_steering_demo --model models/gguf/Qwen3.5-4B-Q4_K_M.gguf --vector data/moods/Qwen3.5-4B-Q4_K_M/joy_sadness.gguf --max-scale 0.7 --il-start 8 --il-end 24
```

La commande tient sur une seule ligne, valable sous PowerShell comme sous bash (les continuations de ligne diffèrent d'un shell à l'autre).

Les trois derniers réglages sont ceux de la campagne : `--max-scale 0.7` (le défaut est 1.0, trop fort pour ce modèle) et les couches 8 à 24. Le chargement prend une vingtaine de secondes. Ouvrir ensuite <http://127.0.0.1:8000>. Le serveur n'écoute que sur la machine locale (`--host` et `--port` pour changer).

Le moteur tourne sur le CPU : la roue du lock est une roue CPU et `n_gpu_layers` vaut 0. Cela suffit à garder le GPU inutilisé : rien à désactiver, avec ou sans GPU. Variable facultative en ceinture et bretelles (vide) : PowerShell `$env:CUDA_VISIBLE_DEVICES = ""`, bash `export CUDA_VISIBLE_DEVICES=`. Le nombre de threads est celui par défaut de llama-cpp-python (la moitié des cœurs logiques).

Une seule génération à la fois : une seconde requête pendant une génération reçoit 409. `GET /health` et `GET /config` renvoient l'état du serveur.

## Scénario pas à pas

Prompt de départ (pré-rempli) : « Raconte-moi ta journée en quelques phrases. »

1. **Neutre, dose 0.** Générer. Les colonnes Neutre et Steering sont identiques (dose 0 : aucun effet, vérifié pendant la répétition du 2026-11-05). Le modèle répond comme une IA (« En tant qu'intelligence artificielle, je n'ai pas de journée personnelle… »). La colonne Prompt joyeux est déjà exubérante (« Ma journée est une véritable explosion de joie… »).
2. **Steering positif.** Dose +0,7 puis +1. La colonne Steering devient joyeuse (« Bonjour ! Ma journée est super dynamique ! », émojis), sans consigne dans le prompt. À +0,43 l'effet est plus faible ; à +1 il atteint le ton du prompt joyeux. À mettre en regard : le prompt joyeux obtient un ton comparable, mais il cède sous une consigne neutre alors que le steering tient (H4, voir le rapport).
3. **Pôle négatif, avec ses limites.** Dose −0,43 : dose négative la plus propre (1 % de sorties dégénérées) ; le ton devient plat et sombre, avec des refus d'assistant occasionnels (« Je ne possède pas de corps physique… »). Dose −0,71 puis −1 : 19 % puis 49 % de sorties dégénérées mesurées ; à −1, la colonne tourne en boucle (« Je ne suis pas là, ni ailleurs. » répété). À présenter tel quel : c'est la limite du pôle négatif, pas un état du modèle.
4. **Retour à 0** pour montrer la réversibilité : rien n'est modifié dans les poids.

5. **Profil « Impulsion » (pharmacocinétique, H6).** Choisir le profil « Impulsion » : le curseur donne alors la dose du pic, et deux champs règlent la montée (tokens jusqu'au pic, 48 par défaut, comme la mesure) et la demi-vie de la décroissance (32 tokens par défaut). La dose monte linéairement de 0 au pic, puis diminue de moitié à chaque demi-vie. Ce que la mesure montre (sorties de 192 tokens, [`rapport-pharmacocinetique.md`](rapport-pharmacocinetique.md)) : avec la montée par défaut, le ton de la colonne Steering reste proche du neutre, sans montée ni redescente lisible : la dose posée pendant la génération agit plus faiblement et plus tard que la dose posée sur le prompt. Pour montrer une redescente, régler la montée à 0 (dose +1, demi-vie 32) : le pic s'applique aussi au prompt, le ton part joyeux puis se relâche à mesure que la dose baisse. La courbe de dose, elle, suit le profil dans les deux cas. Les colonnes Neutre et Prompt joyeux ne changent pas.

La page dessine aussi la dose appliquée au fil des tokens (colonne Steering). En profil constant, la dose ne varie pas pendant une génération : la courbe est une droite horizontale à la valeur du curseur. En profil impulsion, le serveur envoie la dose réellement appliquée à chaque token (champ `dose` des événements `token` du flux, `profile: "pulse"` dans l'événement `start`) et la courbe trace cette dose : montée, pic, décroissance. Paramètres de `/stream` : `profile` (`constant` par défaut, ou `pulse`), `rise` (tokens jusqu'au pic, 0 à 512), `half_life` (demi-vie en tokens, de 0 exclu à 512) ; `dose` est alors le pic. Le bouton « Arrêter » coupe les trois colonnes.

## Vitesse attendue

Répétition du 2026-11-05 sur la tour (CPU seul, GPU non utilisé, Qwen3.5-4B-Q4_K_M, processus limité à 4 cœurs logiques par affinité, non fournie par la démo : par exemple `start /affinity F` sous Windows (cmd), `taskset -c 0-3` sous Linux) :

- environ 5 tokens par seconde et par colonne (4,6 à 6,3 mesurés), les trois colonnes générant en même temps, soit environ 15 tokens par seconde au total ;
- premier token après 1,3 à 1,7 s ;
- une réponse de 128 tokens : 21 à 26 s pour les trois colonnes ; 9 s pour 40 tokens.

Un portable plus lent ou à moins de cœurs sera proportionnellement plus lent : la vitesse réelle est à confirmer sur le portable de présentation. Réduire `max_tokens` (paramètre de `/stream`, 512 au plus) raccourcit d'autant l'attente.

## Limites

- Le steering mesuré ici couvre une seule paire de pôles (joie, tristesse) sur un seul modèle de cœur ; aucune dose au-delà de +1 (échelle 0,7) n'a été mesurée.
- Sorties dégénérées (boucles, refus d'assistant) à forte dose négative ; voir l'étape 3.
- La génération est gloutonne : le même prompt et la même dose redonnent le même texte (la graine n'a pas d'effet).
- Les colonnes peuvent ne pas se terminer au même moment ; une colonne en erreur n'arrête pas les autres.
- Les scores du rapport mesurent le ton du texte, pas un ressenti.

## Repli sans vecteur

Lancer la commande sans `--vector` : la colonne Steering suit alors un prompt d'humeur à la place du vecteur, ce que la page signale par un badge (`fallback` dans l'événement `start`). La démo reste complète en trois colonnes, mais la colonne n'illustre plus le steering d'activations : le dire au public.
