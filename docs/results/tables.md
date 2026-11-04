## France, test 2025

| Modèle | MAPE (%) | RMSE (MW) | MAE (MW) | Biais (MW) |
|---|---|---|---|---|
| LightGBM (météo observée) | 1,61 | 1 139 | 842 | 285 |
| LightGBM (météo prévue à 24 h) | 1,68 | 1 219 | 883 | 317 |
| LightGBM (météo prévue à 48 h) | 1,70 | 1 234 | 893 | 319 |
| RTE J-1 recalée (mois × demi-heure) | 1,72 | 1 164 | 881 | 172 |
| RTE J-1 recalée (écart constant) | 2,13 | 1 352 | 1 061 | 173 |
| RTE, prévision J-1 | 2,67 | 1 656 | 1 324 | -973 |
| Persistance J-7 | 6,41 | 5 139 | 3 451 | -92 |
| Moyenne saisonnière | 6,63 | 4 317 | 3 366 | 2 315 |

### MAPE (%) par saison, France

| | LightGBM (météo prévue à 48 h) | LightGBM (météo observée) | RTE, prévision J-1 | RTE J-1 recalée (écart constant) | RTE J-1 recalée (mois × demi-heure) | Persistance J-7 |
|---|---|---|---|---|---|---|
| hiver | 2,29 | 2,14 | 2,14 | 2,01 | 1,99 | 8,81 |
| printemps | 1,76 | 1,57 | 3,29 | 2,10 | 1,63 | 7,95 |
| été | 1,22 | 1,18 | 2,56 | 2,26 | 1,62 | 4,07 |
| automne | 1,55 | 1,55 | 2,68 | 2,13 | 1,65 | 4,85 |

### Backtest, MAPE (%) par année, France (température observée)

| | LightGBM, cible en MW | LightGBM, cible ratio | RTE, prévision J-1 | RTE J-1 recalée (écart constant) | RTE J-1 recalée (mois × demi-heure) | Persistance J-7 | Moyenne saisonnière |
|---|---|---|---|---|---|---|---|
| 2019 | 1,38 | 1,61 | 1,46 | 1,49 | 1,36 | 6,00 | 5,09 |
| 2020 | 2,52 | 2,50 | 1,60 | 1,60 | 1,59 | 6,01 | 8,60 |
| 2021 | 1,59 | 1,83 | 1,62 | 1,58 | 1,54 | 7,10 | 5,50 |
| 2022 | 1,80 | 1,94 | 1,90 | 1,76 | 1,74 | 6,50 | 7,11 |
| 2023 | 2,03 | 2,04 | 2,68 | 2,04 | 1,98 | 6,48 | 8,48 |
| 2024 | 1,69 | 1,81 | 2,73 | 1,95 | 1,82 | 6,05 | 7,90 |

## Nouvelle-Aquitaine, test 2025

| Modèle | MAPE (%) | RMSE (MW) | MAE (MW) | Biais (MW) |
|---|---|---|---|---|
| LightGBM (météo observée) | 2,64 | 160 | 124 | 33 |
| LightGBM (météo prévue à 24 h) | 2,78 | 172 | 131 | 33 |
| LightGBM (météo prévue à 48 h) | 2,83 | 175 | 134 | 33 |
| Moyenne saisonnière | 8,16 | 502 | 389 | 167 |
| Persistance J-7 | 8,89 | 656 | 447 | -11 |

### MAPE (%) par saison, Nouvelle-Aquitaine

| | LightGBM (météo prévue à 48 h) | LightGBM (météo observée) | Persistance J-7 |
|---|---|---|---|
| hiver | 3,05 | 2,74 | 12,98 |
| printemps | 3,38 | 3,03 | 10,82 |
| été | 2,37 | 2,31 | 5,64 |
| automne | 2,50 | 2,49 | 6,17 |

### Backtest, MAPE (%) par année, Nouvelle-Aquitaine (température observée)

| | LightGBM, cible en MW | Persistance J-7 | Moyenne saisonnière |
|---|---|---|---|
| 2019 | 2,15 | 7,80 | 6,34 |
| 2020 | 2,51 | 7,97 | 8,79 |
| 2021 | 2,21 | 8,92 | 6,90 |
| 2022 | 3,04 | 8,90 | 9,41 |
| 2023 | 3,50 | 8,52 | 8,68 |
| 2024 | 2,85 | 8,26 | 8,71 |

