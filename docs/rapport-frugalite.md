# Rapport de frugalité (H7)

> **Deux mesures : mesure H7 répétée (A4) sur Qwen3.5-4B-Q4_K_M en GPU, 7 tours par bras, puis, plus bas, mesure courte sur Qwen3.5-0.8B-Q4_K_M en CPU, n réduit.** Pas de verdict H7 : seuils non pré-enregistrés (action A6). Réductions de la mesure courte de protocole : sorties de ton limitées à 64 tokens (cible 128), 10 prompts de ton (cible 100), 10 QCM (cible 200), modèles secondaires sans les phases tone_neutral, mcq, syco, syco_fixed. Ordre de grandeur seulement, pas un verdict. Expression d'humeur, pas ressenti.

<!-- lecture -->
**Lecture (rédigée à la main, conservée à la régénération ; tout ce qui précède la section « H7 : coût du steering » est dans ce bloc).** Conditions de la mesure courte du 0,8B : CPU partagé avec un autre agent (mesure 6.2 sur GPU et sa préparation), donc machine qui n'était pas au repos ; n = 10 générations par bras, une seule lecture CodeCarbon par bras, bras mesurés l'un après l'autre, si bien que leur ordre se confond avec la dérive de la charge.

- Énergie et CO2 (0,8B, CPU) : **pas d'écart détectable à cette résolution**. Les ratios inférieurs à 1 sont des lectures uniques, sans intervalle, et ne montrent pas une économie du steering.
- Latence : l'écart apparié par item exclut zéro pour chaque bras de steering, qui va de 0,05 à 0,14 s plus vite par génération que le prompt joyeux. C'est cohérent avec le prompt système du bras joyeux, plus long à traiter, mais avec une machine chargée et un ordre des bras non alterné, la mesure ne suffit pas à l'établir.

Une conclusion demanderait une mesure sur une machine au repos, avec plusieurs lectures par bras et un ordre des bras alterné. La mesure A4 ci-dessous répond à cette condition de mesure sur le 4B (plusieurs lectures par bras, ordre alterné, GPU sans autre charge au lancement), sans donner de verdict.

## Méthode de la mesure A4 (appliquée)

La mesure courte du 0,8B (section H7 ci-dessous) n'a qu'une lecture par bras, sur CPU partagé. L'action A4 de la rétro de l'epic 5 la reprend sur le modèle cœur (4B), en GPU, avec K lectures par bras et l'ordre des bras alterné. Commande lancée (chemins relatifs au dépôt ; modèles, vecteurs et sorties hors git) :

```
uv run python scripts/run_bench.py --models 4b --tone-only --energy --energy-repeats 7 --n-gpu-layers -1 --cuda-dll-dir models/cuda-runtime --out data/runs/energie-a4 --no-judge
```

- `--tone-only` : phase ton seule pour tous les modèles, cœur compris (`--short-run` refuse le cœur) ; ni QCM, ni juge, ni classifieur, et le rapport de frugalité est écrit seul, par défaut dans `{out}/frugalite.md` (`docs/rapport-verdict.md` refusé). Le plancher des 100 QCM ne s'applique pas, faute de QCM ; ceux du ton restent. Reprise : même commande avec `--resume` (K, règle d'ordre et device sont vérifiés).
- Tour 0 = génération notée (mêmes sorties qu'une mesure à K = 1) et chauffe, hors statistiques d'énergie ; les tours 1 à K-1 régénèrent les mêmes 100 prompts de ton sans notation (`{modèle}/tone_r{r}/{bras}.jsonl`) et ne servent qu'à l'énergie et à la latence. Une lecture CodeCarbon par bras et par tour, dans `energy_units.json` (clé `tone#r{r}` pour r > 0). Un tour commencé sans lecture (arrêt brutal) est remesuré en entier à la reprise ; un tour sans lecture est signalé au rapport.
- Ordre des bras alterné à partir du tour 1 : tours impairs dans l'ordre des bras (neutre, prompt joyeux, steering), tours pairs dans l'ordre inverse. K doit être impair (nombre pair de tours mesurés, position moyenne de chaque bras équilibrée). La règle et K sont figés au manifeste ; le relevé de charge au lancement (utilisation GPU par `nvidia-smi`, charge CPU) est dans `runs.jsonl`.
- Rapport : par bras, énergie et CO2 par génération (moyenne, IC95 de Student à K-2 ddl sur les K-1 tours mesurés : le bootstrap percentile sous-couvre sur si peu de tours) et ratio steering / `prompt_joyful` apparié par tour, même IC. Pas de verdict : les seuils de H7 ne sont pas pré-enregistrés (action A6).
- Conditions réelles : lancement le 2026-11-06 à 00 h 23, fin à 03 h 27, sur un portable RTX 3050 (toutes les couches en GPU). 8 bras (neutre, prompt joyeux, 3 doses positives, 3 doses négatives) x 7 tours = 56 unités de mesure, 100 générations de ton par bras et par tour (`n_tone` = 100 au manifeste, 128 jetons au plus). Relevé de charge au lancement : GPU 0 % (0 MiB), CPU 4 %.
- Prérequis : `uv sync --extra frugal`, GPU sans autre charge, vecteur du 4B dans `data/moods`.

## Résultats de la mesure A4 (Qwen3.5-4B, GPU)

Source : rapport brut `frugalite.md` du dossier de sortie de la mesure (hors git). Énergie par génération exprimée en Wh (kWh x 1000) ; moyennes et IC95 de Student (t à 5 ddl) sur les tours 1 à 6, le tour 0 (chauffe, sorties notées) n'y entre pas.

Énergie et CO2 par génération, tours 1 à 6 (CO2 en mg, kg x 1e6) :

| Bras | Énergie/gén. (Wh) | IC95 (Wh) | CO2/gén. (mg) | IC95 (mg) |
|---|---|---|---|---|
| neutral | 0,02952 | [0,02947 ; 0,02957] | 1,654 | [1,652 ; 1,657] |
| prompt_joyful | 0,02920 | [0,02911 ; 0,02929] | 1,636 | [1,631 ; 1,641] |
| steering_p0.43 | 0,02784 | [0,02777 ; 0,02791] | 1,560 | [1,556 ; 1,564] |
| steering_p0.71 | 0,02571 | [0,02566 ; 0,02575] | 1,441 | [1,438 ; 1,443] |
| steering_p1.00 | 0,02221 | [0,02212 ; 0,02229] | 1,244 | [1,240 ; 1,249] |
| steering_m0.43 | 0,03169 | [0,03159 ; 0,03178] | 1,776 | [1,771 ; 1,781] |
| steering_m0.71 | 0,03109 | [0,03102 ; 0,03116] | 1,742 | [1,738 ; 1,746] |
| steering_m1.00 | 0,03139 | [0,03129 ; 0,03149] | 1,759 | [1,754 ; 1,765] |

Ratio d'énergie par génération face à `prompt_joyful`, apparié par tour (moyenne et IC95 de Student des 6 ratios ; le ratio de CO2 est identique) :

| Bras | Ratio | IC95 |
|---|---|---|
| steering_p0.43 | x0,9536 | [0,9508 ; 0,9565] |
| steering_p0.71 | x0,8804 | [0,8776 ; 0,8832] |
| steering_p1.00 | x0,7605 | [0,7568 ; 0,7643] |
| steering_m0.43 | x1,085 | [1,082 ; 1,089] |
| steering_m0.71 | x1,065 | [1,061 ; 1,069] |
| steering_m1.00 | x1,075 | [1,071 ; 1,079] |

Latence, tour 0 seul (100 générations par bras, moyenne et IC95 bootstrap ; le surcoût est le ratio et la différence face à `prompt_joyful`, et l'écart apparié par item exclut zéro pour chaque bras de steering) :

| Bras | Latence (s/gén.) | Surcoût de latence | Latence par mot (s/mot) | IC95 (s/mot) |
|---|---|---|---|---|
| neutral | 2,025 [1,954 ; 2,096] | - | 0,02657 | [0,02617 ; 0,02702] |
| prompt_joyful | 2,000 [1,924 ; 2,072] | référence | 0,02577 | [0,02539 ; 0,02619] |
| steering_p0.43 | 1,902 [1,809 ; 1,992] | x0,9511 ; -0,098 s | 0,02678 | [0,02631 ; 0,02726] |
| steering_p0.71 | 1,759 [1,651 ; 1,862] | x0,8792 ; -0,242 s | 0,02803 | [0,02701 ; 0,02943] |
| steering_p1.00 | 1,514 [1,391 ; 1,639] | x0,757 ; -0,486 s | 0,02942 | [0,02853 ; 0,03039] |
| steering_m0.43 | 2,171 [2,127 ; 2,212] | x1,085 ; +0,170 s | 0,02565 | [0,02531 ; 0,02600] |
| steering_m0.71 | 2,128 [2,072 ; 2,178] | x1,064 ; +0,128 s | 0,02518 | [0,02487 ; 0,02550] |
| steering_m1.00 | 2,150 [2,095 ; 2,200] | x1,075 ; +0,150 s | 0,02349 | [0,02317 ; 0,02382] |

### Lecture

Ces points sont une lecture des chiffres ci-dessus, pas un verdict.

- Les ratios d'énergie suivent la latence, donc la longueur des sorties : les doses positives (p0.43 à p1.00) produisent des sorties plus courtes et consomment moins (x0,95 à x0,76), les doses négatives des sorties plus longues et consomment plus (x1,065 à x1,085). Longueur moyenne au tour 0, en mots par génération : 52,6 à p1.00, 64,5 à p0.71, 70,9 à p0.43, 77,8 pour `prompt_joyful`, 76,4 pour neutral, 84,9 à m0.43, 85,0 à m0.71, 92,3 à m1.00. Ce n'est pas une économie du steering. La latence par mot est plus élevée sous steering positif (0,0294 s/mot à p1.00 contre 0,0258 pour `prompt_joyful`, soit environ +14 %, au tour 0 seul) ; un coût fixe par génération réparti sur des sorties plus courtes pourrait l'expliquer, ce qui n'est pas établi. Le comptage par mots est une approximation (pas de `n_tokens` dans le JSONL).
- Les IC des tours 1 à 6 sont très étroits : ils mesurent la répétabilité de la machine, pas l'incertitude d'échantillonnage des items. Les mêmes prompts sont régénérés à chaque tour, et les sorties des tours 1 à 6 sont identiques à celles du tour 0 pour les 100 prompts de chacun des 8 bras (génération déterministe) : la durée totale d'un bras ne varie que de 0,2 à 0,9 s sur les 7 tours, tour 0 compris.
- Le tour 0 consomme plus que les tours suivants (neutral : 3,45e-05 kWh/gén. au tour 0 contre 2,95e-05 ensuite, soit environ +17 %), alors que la durée de génération du tour 0 est presque la même (neutral : 202,5 s pour 100 générations contre 201,6 à 201,9 s ensuite) : la puissance moyenne lue au tour 0 est donc plus élevée. Cause non établie : chauffe ou charge CPU simultanée (point suivant). L'écart varie selon les bras (de +3 % à +17 %). L'exclusion du tour 0 des statistiques, prévue par le protocole, écarte les deux causes.
- Bruit CPU : pendant le tour 0 (environ 00 h 23 à 00 h 50), la revue d'une autre story (A2) a fait tourner la suite de tests sur la même machine (de 00 h 30 à 00 h 45 environ, d'après le journal du run). Ce bruit touche le tour 0, donc le tableau de latence ci-dessus et l'énergie du tour 0 (rapport brut), pas les tours 1 à 6 qui fondent les statistiques d'énergie.
- Pas de verdict H7 : les seuils des hypothèses bonus ne sont pas pré-enregistrés (action A6, décision d'Anaël). Ce rapport donne les mesures et leurs intervalles.
<!-- /lecture -->

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
