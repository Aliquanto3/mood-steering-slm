# Rapport de frugalité (H7)

> **Mesure courte, Qwen3.5-0.8B-Q4_K_M, CPU, n réduit.** Réductions de protocole : sorties de ton limitées à 64 tokens (cible 128), 10 prompts de ton (cible 100), 10 QCM (cible 200), modèles secondaires sans les phases tone_neutral, mcq, syco, syco_fixed. Ordre de grandeur seulement, pas un verdict. Expression d'humeur, pas ressenti.

## H7 : coût du steering face au prompt joyeux

#### Qwen3.5-0.8B-Q4_K_M

Latence par génération (moyenne, IC95 bootstrap), énergie et CO2 par bras.

| Bras | n | Latence (s) | Latence/mot (approx. tokens) (s) | Énergie (kWh) | CO2 (kg) | Énergie/gén. (kWh) | CO2/gén. (kg) |
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


Phase ton seulement (mêmes items pour tous les bras).
