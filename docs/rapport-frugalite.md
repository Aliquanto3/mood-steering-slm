# Rapport de frugalité (H7)

> **Mesure courte, Qwen3.5-0.8B-Q4_K_M, CPU, n réduit.** Réductions de protocole : sorties de ton limitées à 64 tokens (cible 128), 10 prompts de ton (cible 100), 10 QCM (cible 200), modèles secondaires sans les phases tone_neutral, mcq, syco, syco_fixed. Ordre de grandeur seulement, pas un verdict. Expression d'humeur, pas ressenti.

<!-- lecture -->
**Lecture (rédigée à la main, conservée à la régénération).** Conditions de cette mesure : CPU partagé avec un autre agent (mesure 6.2 sur GPU et sa préparation), donc machine qui n'était pas au repos ; n = 10 générations par bras, une seule lecture CodeCarbon par bras, bras mesurés l'un après l'autre, si bien que leur ordre se confond avec la dérive de la charge.

- Énergie et CO2 : **pas d'écart détectable à cette résolution**. Les ratios inférieurs à 1 sont des lectures uniques, sans intervalle, et ne montrent pas une économie du steering.
- Latence : l'écart apparié par item exclut zéro pour chaque bras de steering, qui va de 0,05 à 0,14 s plus vite par génération que le prompt joyeux. C'est cohérent avec le prompt système du bras joyeux, plus long à traiter, mais avec une machine chargée et un ordre des bras non alterné, la mesure ne suffit pas à l'établir.

Une conclusion demanderait une mesure sur une machine au repos, avec plusieurs lectures par bras et un ordre des bras alterné.
<!-- /lecture -->

## Méthode de la mesure A4 (prévue, non encore lancée)

La mesure ci-dessous n'a qu'une lecture par bras, sur CPU partagé. L'action A4 de la rétro de l'epic 5 la reprend sur le modèle cœur (4B), en GPU, machine au repos, avec K lectures par bras et l'ordre des bras alterné :

```
uv run python scripts/run_bench.py --models 4b --tone-only --energy --energy-repeats 5   --n-gpu-layers -1 --cuda-dll-dir models/cuda-runtime   --out data/runs/energie-a4 --report data/runs/energie-a4/frugalite.md --no-judge
```

- `--tone-only` : phase ton seule pour tous les modèles, cœur compris (`--short-run` refuse le cœur) ; ni QCM, ni juge, ni classifieur, et le rapport de frugalité est écrit seul (`--report` évite d'écraser ce fichier-ci). Reprise : même commande avec `--resume` (K, règle d'ordre et device sont vérifiés).
- Tour 0 = génération notée (mêmes sorties qu'une mesure à K = 1) ; les tours 1 à K-1 régénèrent les mêmes 100 prompts de ton sans notation (`{modèle}/tone_r{r}/{bras}.jsonl`) et ne servent qu'à l'énergie et à la latence. Une lecture CodeCarbon par bras et par tour, dans `energy_units.json` (clé `tone#r{r}` pour r > 0).
- Ordre des bras alterné : tour pair dans l'ordre des bras (neutre, prompt joyeux, steering), tour impair dans l'ordre inverse. La règle et K sont figés au manifeste ; le relevé de charge au lancement (utilisation GPU par `nvidia-smi`, charge CPU) est dans `runs.jsonl`.
- Rapport : par bras, énergie et CO2 par génération (moyenne, IC95 bootstrap sur les K tours) et ratio steering / `prompt_joyful` apparié par tour. Pas de verdict : les seuils de H7 ne sont pas pré-enregistrés (action A6).
- Durée estimée : 8 bras x 5 tours = 40 passes de 100 générations de 128 jetons au plus. Sur la seule latence connue (0,8B, CPU : 1,3 s par génération de 64 jetons, soit une cinquantaine de jetons par seconde), un 4B en GPU à 40-60 jetons par seconde donne 3 à 5 min par passe, soit environ 2 à 3,5 h. Le premier tour affiche la durée réelle de chaque bras ; `--n-tone 50` divise la durée par deux (plancher du protocole).
- Prérequis : `uv sync --extra frugal`, GPU sans autre charge (fermer toute autre mesure), vecteur du 4B dans `data/moods`.

## H7 : coût du steering face au prompt joyeux

#### Qwen3.5-0.8B-Q4_K_M

Latence par génération (moyenne, IC95 bootstrap), énergie et CO2 par bras.

| Bras | n | Latence (s) | Latence par mot (s/mot) | Énergie (kWh) | CO2 (kg) | Énergie/gén. (kWh) | CO2/gén. (kg) |
|---|---|---|---|---|---|---|---|
| neutral | 10 | 1.342 [1.295 ; 1.397] | 0.03084 [0.02682 ; 0.03553] | 0.0003703 | 2.075e-05 | 3.703e-05 | 2.075e-06 |
| prompt_joyful | 10 | 1.382 [1.358 ; 1.409] | 0.02943 [0.02847 ; 0.03042] | 0.0003822 | 2.142e-05 | 3.822e-05 | 2.142e-06 |
| steering_m0.67 | 10 | 1.305 [1.281 ; 1.335] | 0.02827 [0.02624 ; 0.03075] | 0.0003601 | 2.018e-05 | 3.601e-05 | 2.018e-06 |
| steering_m0.83 | 10 | 1.325 [1.309 ; 1.341] | 0.02684 [0.02564 ; 0.02803] | 0.0003649 | 2.045e-05 | 3.649e-05 | 2.045e-06 |
| steering_m1.00 | 10 | 1.327 [1.313 ; 1.342] | 0.02685 [0.02545 ; 0.02843] | 0.0003691 | 2.069e-05 | 3.691e-05 | 2.069e-06 |
| steering_p0.67 | 10 | 1.238 [1.113 ; 1.308] | 0.0305 [0.02803 ; 0.03266] | 0.0003409 | 1.91e-05 | 3.409e-05 | 1.91e-06 |
| steering_p0.83 | 10 | 1.238 [1.126 ; 1.303] | 0.0345 [0.03063 ; 0.03966] | 0.0003402 | 1.907e-05 | 3.402e-05 | 1.907e-06 |
| steering_p1.00 | 10 | 1.295 [1.275 ; 1.317] | 0.03473 [0.03188 ; 0.03701] | 0.0003564 | 1.997e-05 | 3.564e-05 | 1.997e-06 |

Surcoût du steering face à `prompt_joyful` (ratio ; différence) :

| Bras | Latence | Énergie/gén. | CO2/gén. |
|---|---|---|---|
| steering_m0.67 | x0.9447 ; -0.07643 s | x0.9422 ; -2.209e-06 kWh | x0.9422 ; -1.238e-07 kg |
| steering_m0.83 | x0.959 ; -0.0567 s | x0.9548 ; -1.726e-06 kWh | x0.9548 ; -9.673e-08 kg |
| steering_m1.00 | x0.9603 ; -0.0548 s | x0.9659 ; -1.302e-06 kWh | x0.9659 ; -7.298e-08 kg |
| steering_p0.67 | x0.8958 ; -0.144 s | x0.8921 ; -4.125e-06 kWh | x0.8921 ; -2.311e-07 kg |
| steering_p0.83 | x0.8956 ; -0.1442 s | x0.8903 ; -4.192e-06 kWh | x0.8903 ; -2.349e-07 kg |
| steering_p1.00 | x0.9375 ; -0.0864 s | x0.9326 ; -2.576e-06 kWh | x0.9326 ; -1.444e-07 kg |

Limite de mesure : CodeCarbon estime la consommation de la machine entière (CPU, GPU, mémoire) à partir de modèles et de facteurs d'émission du pays : c'est une estimation, pas une mesure au wattmètre. Une autre charge simultanée sur la machine fausse la mesure ; les bras doivent être mesurés sur une machine par ailleurs au repos.
Les tokens sont comptés par mots de la sortie (approximation) en l'absence de `n_tokens` dans le JSONL.

Écart de latence apparié par item face à `prompt_joyful` (s, IC95 bootstrap) :

| Bras | n | Écart (s) | Lecture |
|---|---|---|---|
| steering_p0.67 | 10 | -0.144 [-0.2718 ; -0.06473] | IC hors de zéro |
| steering_p0.83 | 10 | -0.1442 [-0.2546 ; -0.07144] | IC hors de zéro |
| steering_p1.00 | 10 | -0.0864 [-0.1034 ; -0.0685] | IC hors de zéro |
| steering_m0.67 | 10 | -0.07643 [-0.1165 ; -0.0361] | IC hors de zéro |
| steering_m0.83 | 10 | -0.0567 [-0.09206 ; -0.02616] | IC hors de zéro |
| steering_m1.00 | 10 | -0.0548 [-0.07461 ; -0.03459] | IC hors de zéro |

Énergie et CO2 : une seule lecture par bras, sans intervalle ; les bras sont mesurés l'un après l'autre, si bien que leur ordre se confond avec la dérive de la machine. Un ratio d'énergie proche de 1 ne vaut pas économie : seul un écart de latence dont l'IC exclut zéro, mesuré sur une machine au repos, permettrait de conclure.


Phase ton seulement (mêmes items pour tous les bras).
