# E8-07 -- evidencia extendida de la politica v6 (KPI refinado), comparada con v5

Muestra: 362 batches, identica para v5 y v6 (misma union DEV/lockbox, 63 lockbox / 299 DEV). Replica la metodologia de E7-06 sobre la politica v6.

## Resumen ejecutivo

- **Cadena theta -> KPI (C)**: la proyeccion interna (PLM + OLS de batch, controles E8-03) da un uplift ESPERADO **mayor** para v6 que
  para v5 -- DEV: 0.262 pp [0.175, 0.359] (~134 kg Sn/batch) vs 0.206 pp [0.128, 0.312] (~106 kg Sn/batch); TOTAL: 0.281 pp (~144 kg)
  vs 0.221 pp (~113 kg). Es coherente con que `m6_ln_feo_ret` tiene menos ruido de ensayo que `v5_ln_feo_ret` (ITERACION_8_diseno.md).
  Pero esta cifra es un calculo **interno** de cada sistema (su propio PLM cross-fitted + su propia pendiente OLS), no evidencia
  independiente de que la politica funcione fuera de muestra.
- **Adherencia vs KPI (out-of-sample, ya reportada en E8-05/E7-05) y emparejamiento por estado (A, nuevo aqui)**: la evidencia
  off-policy directa es **igual o mas debil** para v6 que para v5, sobre todo en Fusión (ver Veredicto).
- **Comparacion v5 vs v6 (D)**: ambas politicas coinciden bastante en que' batches se desvian (r 0.86-0.92 en dist) y en el uplift
  fisico esperado (r 0.79-0.97), y coinciden en el signo de casi todas las palancas de Fusión (86-99.7 %) y en aire/GN/O2 de
  Reduccion (76-78 %), pero **discrepan fuertemente en el carbon de Reduccion** (51.7 % de coincidencia de signo, correlacion 0.27) --
  la palanca central de la hipotesis de sobre-reduccion.

## A) Emparejamiento por estado -- v6 (cerca vs lejos de la politica; dif > 0 favorece seguirla)

      dist                              kpi  n_pares  dif_media_pp  ci_lo  ci_hi  p_pareado  smd_medio_post  dist_media_cerca  dist_media_lejos  subset sistema
dist_total        recuperacion_refinada_pct       99        -1.204 -2.319 -0.124      0.035           0.054             0.133             0.406     DEV      v6
dist_total          rendimiento_proxy_batch       99        -1.428 -2.534 -0.237      0.016           0.054             0.133             0.406     DEV      v6
dist_total   recuperacion_refinada_pct_next       99         0.046 -1.086  1.115      0.938           0.054             0.133             0.406     DEV      v6
dist_total recuperacion_refinada_pct_media2       99        -0.579 -1.356  0.231      0.176           0.054             0.133             0.406     DEV      v6
    dist_R        recuperacion_refinada_pct       99         0.723 -0.367  1.877      0.217           0.035             0.098             0.443     DEV      v6
    dist_R          rendimiento_proxy_batch       99         0.620 -0.603  1.743      0.303           0.035             0.098             0.443     DEV      v6
    dist_R   recuperacion_refinada_pct_next       99         1.535  0.423  2.722      0.012           0.035             0.098             0.443     DEV      v6
    dist_R recuperacion_refinada_pct_media2       99         1.129  0.315  1.962      0.010           0.035             0.098             0.443     DEV      v6
    dist_F        recuperacion_refinada_pct       99         0.273 -0.960  1.450      0.647           0.037             0.108             0.469     DEV      v6
    dist_F          rendimiento_proxy_batch       99         0.262 -0.901  1.443      0.669           0.037             0.108             0.469     DEV      v6
    dist_F   recuperacion_refinada_pct_next       99         0.921 -0.261  2.180      0.150           0.037             0.108             0.469     DEV      v6
    dist_F recuperacion_refinada_pct_media2       99         0.597 -0.258  1.414      0.153           0.037             0.108             0.469     DEV      v6
dist_total        recuperacion_refinada_pct       21        -0.984 -3.256  1.046      0.386           0.119             0.251             0.496 LOCKBOX      v6
dist_total          rendimiento_proxy_batch       21        -0.564 -2.248  1.319      0.561           0.119             0.251             0.496 LOCKBOX      v6
dist_total   recuperacion_refinada_pct_next       21         2.802  0.154  5.216      0.047           0.120             0.252             0.496 LOCKBOX      v6
dist_total recuperacion_refinada_pct_media2       21         1.540  0.201  2.978      0.043           0.120             0.252             0.496 LOCKBOX      v6
    dist_R        recuperacion_refinada_pct       21        -2.821 -5.695 -0.285      0.054           0.129             0.148             0.466 LOCKBOX      v6
    dist_R          rendimiento_proxy_batch       21        -2.200 -4.697  0.303      0.108           0.129             0.148             0.466 LOCKBOX      v6
    dist_R   recuperacion_refinada_pct_next       21         3.503  1.394  6.055      0.012           0.111             0.149             0.466 LOCKBOX      v6
    dist_R recuperacion_refinada_pct_media2       21         0.516 -1.007  2.059      0.544           0.111             0.149             0.466 LOCKBOX      v6
    dist_F        recuperacion_refinada_pct       21        -0.051 -1.637  1.554      0.953           0.120             0.267             0.625 LOCKBOX      v6
    dist_F          rendimiento_proxy_batch       21        -0.073 -1.705  1.537      0.933           0.120             0.267             0.625 LOCKBOX      v6
    dist_F   recuperacion_refinada_pct_next       21         2.655  0.167  5.240      0.059           0.120             0.267             0.625 LOCKBOX      v6
    dist_F recuperacion_refinada_pct_media2       21         1.302  0.056  2.585      0.066           0.120             0.267             0.625 LOCKBOX      v6
dist_total        recuperacion_refinada_pct      120         0.375 -0.668  1.413      0.481           0.049             0.141             0.435   TOTAL      v6
dist_total          rendimiento_proxy_batch      120         0.298 -0.751  1.358      0.584           0.049             0.141             0.435   TOTAL      v6
dist_total   recuperacion_refinada_pct_next      120         3.147  2.083  4.354      0.000           0.049             0.141             0.435   TOTAL      v6
dist_total recuperacion_refinada_pct_media2      120         1.701  0.819  2.542      0.000           0.049             0.141             0.435   TOTAL      v6
    dist_R        recuperacion_refinada_pct      120        -0.004 -0.921  1.042      0.994           0.025             0.103             0.450   TOTAL      v6
    dist_R          rendimiento_proxy_batch      120        -0.095 -1.029  0.873      0.852           0.025             0.103             0.450   TOTAL      v6
    dist_R   recuperacion_refinada_pct_next      120         0.941 -0.032  2.017      0.071           0.025             0.103             0.450   TOTAL      v6
    dist_R recuperacion_refinada_pct_media2      120         0.469 -0.266  1.245      0.212           0.025             0.103             0.450   TOTAL      v6
    dist_F        recuperacion_refinada_pct      120         0.565 -0.410  1.443      0.224           0.036             0.118             0.522   TOTAL      v6
    dist_F          rendimiento_proxy_batch      120         0.603 -0.334  1.546      0.204           0.036             0.118             0.522   TOTAL      v6
    dist_F   recuperacion_refinada_pct_next      120         3.737  2.635  4.782      0.000           0.036             0.118             0.522   TOTAL      v6
    dist_F recuperacion_refinada_pct_media2      120         2.095  1.392  2.821      0.000           0.036             0.118             0.522   TOTAL      v6

## B) Dosis-respuesta por terciles de adherencia -- v6 (KPI refinado, f_dross, f_polvo)

      dist                    metric   n   cerca   medio   lejos  d_cerca_lejos  ols_coef_por_sd  ols_p  subset sistema
dist_total recuperacion_refinada_pct 297 68.3327 68.1999 67.8306         0.5021          -0.0512 0.8339     DEV      v6
dist_total                   f_dross 297  0.1283  0.1305  0.1240         0.0043          -0.0037 0.0893     DEV      v6
dist_total                   f_polvo 297  0.1785  0.1776  0.1869        -0.0085           0.0035 0.0901     DEV      v6
    dist_R recuperacion_refinada_pct 297 68.3800 67.9875 67.9958         0.3842           0.2085 0.3532     DEV      v6
    dist_R                   f_dross 297  0.1292  0.1275  0.1261         0.0032          -0.0031 0.1464     DEV      v6
    dist_R                   f_polvo 297  0.1774  0.1826  0.1830        -0.0057           0.0001 0.9394     DEV      v6
    dist_F recuperacion_refinada_pct 297 68.7066 68.2426 67.4141         1.2925          -0.3984 0.1543     DEV      v6
    dist_F                   f_dross 297  0.1267  0.1316  0.1244         0.0023          -0.0021 0.3100     DEV      v6
    dist_F                   f_polvo 297  0.1752  0.1768  0.1910        -0.0158           0.0062 0.0078     DEV      v6
dist_total recuperacion_refinada_pct  63 68.7193 69.2540 69.9848        -1.2655          -0.6542 0.1893 LOCKBOX      v6
dist_total                   f_dross  63  0.1185  0.1203  0.1076         0.0109           0.0038 0.4532 LOCKBOX      v6
dist_total                   f_polvo  63  0.1819  0.1778  0.1833        -0.0013           0.0036 0.4422 LOCKBOX      v6
    dist_R recuperacion_refinada_pct  63 68.9934 69.0284 69.9363        -0.9429          -0.3713 0.5513 LOCKBOX      v6
    dist_R                   f_dross  63  0.1135  0.1157  0.1173        -0.0038           0.0023 0.6148 LOCKBOX      v6
    dist_R                   f_polvo  63  0.1838  0.1852  0.1740         0.0098           0.0025 0.6507 LOCKBOX      v6
    dist_F recuperacion_refinada_pct  63 68.7714 69.0469 70.1399        -1.3685          -0.6187 0.2621 LOCKBOX      v6
    dist_F                   f_dross  63  0.1233  0.1181  0.1049         0.0184           0.0034 0.5713 LOCKBOX      v6
    dist_F                   f_polvo  63  0.1789  0.1795  0.1846        -0.0056           0.0030 0.5171 LOCKBOX      v6
dist_total recuperacion_refinada_pct 360 68.4554 68.0262 68.5108        -0.0554           0.0452 0.8438   TOTAL      v6
dist_total                   f_dross 360  0.1278  0.1304  0.1182         0.0096          -0.0047 0.0281   TOTAL      v6
dist_total                   f_polvo 360  0.1777  0.1795  0.1858        -0.0081           0.0039 0.0370   TOTAL      v6
    dist_R recuperacion_refinada_pct 360 68.4632 68.0827 68.4465         0.0167           0.1588 0.4572   TOTAL      v6
    dist_R                   f_dross 360  0.1286  0.1233  0.1245         0.0041          -0.0024 0.2194   TOTAL      v6
    dist_R                   f_polvo 360  0.1772  0.1852  0.1806        -0.0034           0.0003 0.8524   TOTAL      v6
    dist_F recuperacion_refinada_pct 360 68.9072 67.3884 68.6968         0.2104          -0.1407 0.5586   TOTAL      v6
    dist_F                   f_dross 360  0.1251  0.1347  0.1165         0.0086          -0.0047 0.0221   TOTAL      v6
    dist_F                   f_polvo 360  0.1750  0.1819  0.1861        -0.0111           0.0062 0.0017   TOTAL      v6

## C) Cadena theta -> agregado -> KPI -- v6 vs v5 (misma muestra, controles E8-03)

sistema pendientes_de                              kpi politica_en  n_batches  b_sn_dep_pp_por_unidad  b_sn_lo  b_sn_hi  b_feo_ret_pp_por_unidad  b_feo_lo  b_feo_hi  w_implicito  d_sn_dep_medio  d_feo_ret_medio  uplift_kpi_pp_medio  uplift_kpi_pp_mediana  uplift_ci_lo  uplift_ci_hi  uplift_kg_sn_medio  uplift_kg_sn_ci_lo  uplift_kg_sn_ci_hi  pct_batches_uplift_pos
     v6           DEV        recuperacion_refinada_pct         DEV        299                   2.853    1.620    4.016                   17.262    11.570    23.449        6.051          -0.006            0.016                0.262                  0.195         0.175         0.359             134.291              89.885             183.847                  98.328
     v6           DEV        recuperacion_refinada_pct     LOCKBOX         63                   2.853    1.620    4.016                   17.262    11.570    23.449        6.051          -0.033            0.027                0.370                  0.334         0.233         0.521             189.462             119.370             266.812                 100.000
     v6           DEV        recuperacion_refinada_pct       TOTAL        362                   2.853    1.620    4.016                   17.262    11.570    23.449        6.051          -0.011            0.018                0.281                  0.221         0.185         0.388             143.893              94.669             199.015                  98.619
     v6           DEV          rendimiento_proxy_batch         DEV        299                   2.396    1.184    3.619                   18.033    12.104    24.441        7.526          -0.006            0.016                0.277                  0.216         0.184         0.377             142.110              94.381             192.980                  97.324
     v6           DEV          rendimiento_proxy_batch     LOCKBOX         63                   2.396    1.184    3.619                   18.033    12.104    24.441        7.526          -0.033            0.027                0.405                  0.378         0.256         0.557             207.742             131.044             285.344                 100.000
     v6           DEV          rendimiento_proxy_batch       TOTAL        362                   2.396    1.184    3.619                   18.033    12.104    24.441        7.526          -0.011            0.018                0.300                  0.240         0.197         0.406             153.532             101.132             208.019                  97.790
     v6           DEV recuperacion_refinada_pct_media2         DEV        299                   1.761    0.984    2.712                   13.839     9.713    18.572        7.858          -0.006            0.016                0.213                  0.169         0.148         0.286             109.300              76.094             146.594                  97.324
     v6           DEV recuperacion_refinada_pct_media2     LOCKBOX         63                   1.761    0.984    2.712                   13.839     9.713    18.572        7.858          -0.033            0.027                0.314                  0.292         0.212         0.426             160.732             108.422             218.280                 100.000
     v6           DEV recuperacion_refinada_pct_media2       TOTAL        362                   1.761    0.984    2.712                   13.839     9.713    18.572        7.858          -0.011            0.018                0.231                  0.185         0.159         0.310             118.251              81.571             159.099                  97.790
     v6         TOTAL        recuperacion_refinada_pct         DEV        299                   3.218    2.091    4.256                   15.708    10.201    21.279        4.881          -0.006            0.016                0.235                  0.166         0.148         0.322             120.267              75.726             165.254                  97.993
     v6         TOTAL        recuperacion_refinada_pct     LOCKBOX         63                   3.218    2.091    4.256                   15.708    10.201    21.279        4.881          -0.033            0.027                0.316                  0.276         0.180         0.448             161.955              92.373             229.854                 100.000
     v6         TOTAL        recuperacion_refinada_pct       TOTAL        362                   3.218    2.091    4.256                   15.708    10.201    21.279        4.881          -0.011            0.018                0.249                  0.183         0.153         0.344             127.522              78.441             176.108                  98.343
     v6         TOTAL          rendimiento_proxy_batch         DEV        299                   2.661    1.614    3.617                   16.326    10.778    21.802        6.136          -0.006            0.016                0.248                  0.184         0.161         0.333             127.124              82.579             170.633                  98.328
     v6         TOTAL          rendimiento_proxy_batch     LOCKBOX         63                   2.661    1.614    3.617                   16.326    10.778    21.802        6.136          -0.033            0.027                0.351                  0.319         0.214         0.484             179.815             109.516             247.924                 100.000
     v6         TOTAL          rendimiento_proxy_batch       TOTAL        362                   2.661    1.614    3.617                   16.326    10.778    21.802        6.136          -0.011            0.018                0.266                  0.209         0.171         0.358             136.294              87.581             183.584                  98.619
     v6         TOTAL recuperacion_refinada_pct_media2         DEV        299                   1.912    1.150    2.806                   13.471     9.603    17.826        7.047          -0.006            0.016                0.206                  0.159         0.146         0.272             105.782              75.022             139.537                  97.993
     v6         TOTAL recuperacion_refinada_pct_media2     LOCKBOX         63                   1.912    1.150    2.806                   13.471     9.603    17.826        7.047          -0.033            0.027                0.299                  0.281         0.207         0.397             153.149             105.963             203.270                 100.000
     v6         TOTAL recuperacion_refinada_pct_media2       TOTAL        362                   1.912    1.150    2.806                   13.471     9.603    17.826        7.047          -0.011            0.018                0.222                  0.178         0.157         0.294             114.026              80.717             150.909                  98.343
     v5           DEV        recuperacion_refinada_pct         DEV        299                   2.124    0.736    3.353                   12.827     8.967    18.291        6.040          -0.028            0.021                0.206                  0.146         0.128         0.312             105.733              65.631             160.156                  96.990
     v5           DEV        recuperacion_refinada_pct     LOCKBOX         63                   2.124    0.736    3.353                   12.827     8.967    18.291        6.040          -0.052            0.031                0.292                  0.243         0.171         0.454             149.683              87.403             232.728                  98.413
     v5           DEV        recuperacion_refinada_pct       TOTAL        362                   2.124    0.736    3.353                   12.827     8.967    18.291        6.040          -0.032            0.023                0.221                  0.164         0.137         0.337             113.382              70.202             172.800                  97.238
     v5           DEV          rendimiento_proxy_batch         DEV        299                   1.569    0.275    2.809                   13.237     9.152    19.286        8.436          -0.028            0.021                0.230                  0.173         0.152         0.350             117.960              78.028             179.532                  97.659
     v5           DEV          rendimiento_proxy_batch     LOCKBOX         63                   1.569    0.275    2.809                   13.237     9.152    19.286        8.436          -0.052            0.031                0.334                  0.286         0.215         0.516             171.089             110.276             264.204                  98.413
     v5           DEV          rendimiento_proxy_batch       TOTAL        362                   1.569    0.275    2.809                   13.237     9.152    19.286        8.436          -0.032            0.023                0.248                  0.189         0.164         0.379             127.207              83.965             194.305                  97.790
     v5           DEV recuperacion_refinada_pct_media2         DEV        299                   1.329    0.479    2.231                    8.676     5.592    13.452        6.526          -0.028            0.021                0.143                  0.102         0.072         0.238              73.042              36.824             121.801                  97.324
     v5           DEV recuperacion_refinada_pct_media2     LOCKBOX         63                   1.329    0.479    2.231                    8.676     5.592    13.452        6.526          -0.052            0.031                0.203                  0.170         0.089         0.346             104.108              45.584             177.495                  98.413
     v5           DEV recuperacion_refinada_pct_media2       TOTAL        362                   1.329    0.479    2.231                    8.676     5.592    13.452        6.526          -0.032            0.023                0.153                  0.118         0.076         0.257              78.449              38.770             131.795                  97.514
     v5         TOTAL        recuperacion_refinada_pct         DEV        299                   2.586    1.492    3.543                   11.915     8.365    16.571        4.607          -0.028            0.021                0.175                  0.123         0.106         0.265              89.493              54.092             135.768                  96.321
     v5         TOTAL        recuperacion_refinada_pct     LOCKBOX         63                   2.586    1.492    3.543                   11.915     8.365    16.571        4.607          -0.052            0.031                0.239                  0.212         0.135         0.377             122.654              69.100             193.403                  96.825
     v5         TOTAL        recuperacion_refinada_pct       TOTAL        362                   2.586    1.492    3.543                   11.915     8.365    16.571        4.607          -0.032            0.023                0.186                  0.134         0.111         0.283              95.264              57.073             145.099                  96.409
     v5         TOTAL          rendimiento_proxy_batch         DEV        299                   1.934    0.838    2.946                   12.211     8.511    17.468        6.313          -0.028            0.021                0.199                  0.142         0.127         0.300             101.899              65.264             153.974                  96.990
     v5         TOTAL          rendimiento_proxy_batch     LOCKBOX         63                   1.934    0.838    2.946                   12.211     8.511    17.468        6.313          -0.052            0.031                0.283                  0.236         0.174         0.433             144.830              89.277             221.862                  98.413
     v5         TOTAL          rendimiento_proxy_batch       TOTAL        362                   1.934    0.838    2.946                   12.211     8.511    17.468        6.313          -0.032            0.023                0.213                  0.161         0.135         0.324             109.370              69.342             165.970                  97.238
     v5         TOTAL recuperacion_refinada_pct_media2         DEV        299                   1.455    0.732    2.432                    8.705     5.885    12.477        5.981          -0.028            0.021                0.140                  0.098         0.078         0.217              71.554              40.060             111.113                  96.656
     v5         TOTAL recuperacion_refinada_pct_media2     LOCKBOX         63                   1.455    0.732    2.432                    8.705     5.885    12.477        5.981          -0.052            0.031                0.197                  0.165         0.098         0.319             101.204              50.222             163.254                  98.413
     v5         TOTAL recuperacion_refinada_pct_media2       TOTAL        362                   1.455    0.732    2.432                    8.705     5.885    12.477        5.981          -0.032            0.023                0.150                  0.111         0.081         0.234              76.714              41.345             119.822                  96.961

## D) Comparacion directa v5 vs v6 por batch

                               metric              tipo   n  pearson_r  pearson_p  spearman_rho  spearman_p  media_v5  media_v6  pct_mismo_signo
                           dist_total          continuo 362      0.896        0.0         0.865         0.0     0.258     0.276              NaN
                               dist_R          continuo 362      0.866        0.0         0.662         0.0     0.221     0.248              NaN
                               dist_F          continuo 360      0.919        0.0         0.926         0.0     0.296     0.307              NaN
               uplift_fisico_total_kg          continuo 362      0.833        0.0         0.763         0.0   181.997   243.318              NaN
              uplift_fisico_fusion_kg          continuo 362      0.969        0.0         0.952         0.0    67.252    83.440              NaN
           uplift_fisico_reduccion_kg          continuo 362      0.791        0.0         0.613         0.0   114.745   159.878              NaN
         dir_Fusión_tasa_aire_nm3_min direccion_palanca 360      0.905        0.0         0.862         0.0    -0.042    -0.038           86.111
   dir_Fusión_tasa_feed_Carbon_kg_min direccion_palanca 360      0.960        0.0         0.949         0.0    -0.484    -0.499           99.722
           dir_Fusión_tasa_gn_nm3_min direccion_palanca 360      0.902        0.0         0.942         0.0    -0.152    -0.133           91.944
           dir_Fusión_tasa_o2_nm3_min direccion_palanca 360      0.916        0.0         0.946         0.0    -0.110    -0.081           92.500
      dir_Reducción_tasa_aire_nm3_min direccion_palanca 362      0.935        0.0         0.682         0.0     0.152     0.240           76.796
dir_Reducción_tasa_feed_Carbon_kg_min direccion_palanca 362      0.273        0.0         0.214         0.0     0.011    -0.126           51.657
        dir_Reducción_tasa_gn_nm3_min direccion_palanca 362      0.776        0.0         0.743         0.0    -0.237    -0.116           77.348
        dir_Reducción_tasa_o2_nm3_min direccion_palanca 362      0.695        0.0         0.664         0.0    -0.089    -0.104           75.967

## E) Politica v6 por tramo de avance (Reduccion) y orden de escalon (Fusion)

        segmento      tramo                 palanca  unidad   n  mediana_fisica  mediana_sd
Reduccion_avance      <0.84 tasa_feed_Carbon_kg_min  kg/min 319           1.475       0.067
Reduccion_avance      <0.84         tasa_gn_nm3_min Nm3/min 319           0.000       0.000
Reduccion_avance      <0.84         tasa_o2_nm3_min Nm3/min 319           0.000       0.000
Reduccion_avance      <0.84       tasa_aire_nm3_min Nm3/min 319           0.000       0.000
Reduccion_avance  0.84-0.96 tasa_feed_Carbon_kg_min  kg/min 232          -3.519      -0.160
Reduccion_avance  0.84-0.96         tasa_gn_nm3_min Nm3/min 232           0.000       0.000
Reduccion_avance  0.84-0.96         tasa_o2_nm3_min Nm3/min 232           0.000       0.000
Reduccion_avance  0.84-0.96       tasa_aire_nm3_min Nm3/min 232           0.453       0.040
Reduccion_avance 0.96-0.975 tasa_feed_Carbon_kg_min  kg/min 365          -4.204      -0.191
Reduccion_avance 0.96-0.975         tasa_gn_nm3_min Nm3/min 365          -0.662      -0.132
Reduccion_avance 0.96-0.975         tasa_o2_nm3_min Nm3/min 365          -0.066      -0.008
Reduccion_avance 0.96-0.975       tasa_aire_nm3_min Nm3/min 365           0.454       0.041
Reduccion_avance     >0.975 tasa_feed_Carbon_kg_min  kg/min 514          -2.529      -0.115
Reduccion_avance     >0.975         tasa_gn_nm3_min Nm3/min 514          -0.662      -0.132
Reduccion_avance     >0.975         tasa_o2_nm3_min Nm3/min 514           0.000       0.000
Reduccion_avance     >0.975       tasa_aire_nm3_min Nm3/min 514           0.452       0.040
    Fusion_orden          1 tasa_feed_Carbon_kg_min  kg/min 357          -2.284      -0.262
    Fusion_orden          1         tasa_gn_nm3_min Nm3/min 357          -0.230      -0.143
    Fusion_orden          1         tasa_o2_nm3_min Nm3/min 357           0.000       0.000
    Fusion_orden          1       tasa_aire_nm3_min Nm3/min 357           0.000       0.000
    Fusion_orden          2 tasa_feed_Carbon_kg_min  kg/min 360          -3.206      -0.368
    Fusion_orden          2         tasa_gn_nm3_min Nm3/min 360           0.000       0.000
    Fusion_orden          2         tasa_o2_nm3_min Nm3/min 360           0.000       0.000
    Fusion_orden          2       tasa_aire_nm3_min Nm3/min 360           0.000       0.000
    Fusion_orden          3 tasa_feed_Carbon_kg_min  kg/min 358          -3.179      -0.365
    Fusion_orden          3         tasa_gn_nm3_min Nm3/min 358           0.000       0.000
    Fusion_orden          3         tasa_o2_nm3_min Nm3/min 358           0.000       0.000
    Fusion_orden          3       tasa_aire_nm3_min Nm3/min 358           0.000       0.000
    Fusion_orden          4 tasa_feed_Carbon_kg_min  kg/min 359          -2.288      -0.263
    Fusion_orden          4         tasa_gn_nm3_min Nm3/min 359           0.000       0.000
    Fusion_orden          4         tasa_o2_nm3_min Nm3/min 359           0.000       0.000
    Fusion_orden          4       tasa_aire_nm3_min Nm3/min 359           0.000       0.000
    Fusion_orden          5 tasa_feed_Carbon_kg_min  kg/min 360          -3.043      -0.350
    Fusion_orden          5         tasa_gn_nm3_min Nm3/min 360           0.000       0.000
    Fusion_orden          5         tasa_o2_nm3_min Nm3/min 360           0.000       0.000
    Fusion_orden          5       tasa_aire_nm3_min Nm3/min 360           0.000       0.000
    Fusion_orden          6 tasa_feed_Carbon_kg_min  kg/min 353          -2.445      -0.281
    Fusion_orden          6         tasa_gn_nm3_min Nm3/min 353           0.000       0.000
    Fusion_orden          6         tasa_o2_nm3_min Nm3/min 353           0.000       0.000
    Fusion_orden          6       tasa_aire_nm3_min Nm3/min 353           0.000       0.000

## Adherencia vs KPI (E8-05 / E7-05, referencia para el veredicto)

sistema                       kpi   variable  subset   n  coef_por_sd  p_hc3  spearman
     v6 recuperacion_refinada_pct dist_total     DEV 298      -0.1455 0.5807   -0.0514
     v6 recuperacion_refinada_pct     dist_F     DEV 297      -0.4241 0.1741   -0.0929
     v6 recuperacion_refinada_pct     dist_R     DEV 298       0.0630 0.7729   -0.0393
     v6                   f_dross dist_total     DEV 298      -0.0033 0.1462   -0.0398
     v6                   f_dross     dist_F     DEV 297      -0.0020 0.4026   -0.0156
     v6                   f_dross     dist_R     DEV 298      -0.0023 0.2504   -0.0179
     v6                   f_polvo dist_total     DEV 298       0.0041 0.0519    0.1443
     v6                   f_polvo     dist_F     DEV 297       0.0063 0.0064    0.1751
     v6                   f_polvo     dist_R     DEV 298       0.0008 0.6247    0.0844
     v6 recuperacion_refinada_pct dist_total LOCKBOX  63      -0.0760 0.8978    0.0863
     v6 recuperacion_refinada_pct     dist_F LOCKBOX  63      -0.0075 0.9902    0.1498
     v6 recuperacion_refinada_pct     dist_R LOCKBOX  63      -0.0998 0.8779   -0.0516
     v6                   f_dross dist_total LOCKBOX  63      -0.0006 0.9120   -0.1256
     v6                   f_dross     dist_F LOCKBOX  63      -0.0012 0.8405   -0.1924
     v6                   f_dross     dist_R LOCKBOX  63       0.0002 0.9644    0.1147
     v6                   f_polvo dist_total LOCKBOX  63       0.0026 0.5753    0.0732
     v6                   f_polvo     dist_F LOCKBOX  63       0.0019 0.6772    0.0755
     v6                   f_polvo     dist_R LOCKBOX  63       0.0020 0.7039   -0.0974
     v6 recuperacion_refinada_pct dist_total   TOTAL 361      -0.0248 0.9196    0.0041
     v6 recuperacion_refinada_pct     dist_F   TOTAL 360      -0.1184 0.6580   -0.0168
     v6 recuperacion_refinada_pct     dist_R   TOTAL 361       0.0131 0.9498   -0.0232
     v6                   f_dross dist_total   TOTAL 361      -0.0044 0.0464   -0.0826
     v6                   f_dross     dist_F   TOTAL 360      -0.0048 0.0331   -0.0836
     v6                   f_dross     dist_R   TOTAL 361      -0.0016 0.3876   -0.0224
     v6                   f_polvo dist_total   TOTAL 361       0.0043 0.0222    0.1157
     v6                   f_polvo     dist_F   TOTAL 360       0.0061 0.0019    0.1491
     v6                   f_polvo     dist_R   TOTAL 361       0.0009 0.5609    0.0611
     v5 recuperacion_refinada_pct dist_total     DEV 298      -0.3367 0.2313   -0.1037
     v5 recuperacion_refinada_pct     dist_F     DEV 297      -0.7032 0.0249   -0.1205
     v5 recuperacion_refinada_pct     dist_R     DEV 298       0.0191 0.9286   -0.1067
     v5                   f_dross dist_total     DEV 298      -0.0022 0.3206    0.0214
     v5                   f_dross     dist_F     DEV 297      -0.0012 0.6197    0.0011
     v5                   f_dross     dist_R     DEV 298      -0.0015 0.4347    0.0846
     v5                   f_polvo dist_total     DEV 298       0.0051 0.0243    0.1354
     v5                   f_polvo     dist_F     DEV 297       0.0085 0.0002    0.1958
     v5                   f_polvo     dist_R     DEV 298       0.0006 0.7233    0.0548
     v5 recuperacion_refinada_pct dist_total LOCKBOX  63       0.0872 0.8765    0.0501
     v5 recuperacion_refinada_pct     dist_F LOCKBOX  63       0.0427 0.9477    0.1204
     v5 recuperacion_refinada_pct     dist_R LOCKBOX  63       0.0849 0.8674   -0.0359
     v5                   f_dross dist_total LOCKBOX  63       0.0017 0.7501   -0.0658
     v5                   f_dross     dist_F LOCKBOX  63      -0.0004 0.9467   -0.1811
     v5                   f_dross     dist_R LOCKBOX  63       0.0027 0.5809    0.0974
     v5                   f_polvo dist_total LOCKBOX  63      -0.0017 0.6561    0.0230
     v5                   f_polvo     dist_F LOCKBOX  63       0.0007 0.8769    0.0828
     v5                   f_polvo     dist_R LOCKBOX  63      -0.0030 0.5151   -0.1107
     v5 recuperacion_refinada_pct dist_total   TOTAL 361      -0.1320 0.5963   -0.0439
     v5 recuperacion_refinada_pct     dist_F   TOTAL 360      -0.2999 0.2810   -0.0482
     v5 recuperacion_refinada_pct     dist_R   TOTAL 361       0.0114 0.9533   -0.0766
     v5                   f_dross dist_total   TOTAL 361      -0.0038 0.0810   -0.0324
     v5                   f_dross     dist_F   TOTAL 360      -0.0045 0.0491   -0.0638
     v5                   f_dross     dist_R   TOTAL 361      -0.0011 0.5633    0.0630
     v5                   f_polvo dist_total   TOTAL 361       0.0049 0.0106    0.1116
     v5                   f_polvo     dist_F   TOTAL 360       0.0078 0.0001    0.1656
     v5                   f_polvo     dist_R   TOTAL 361       0.0004 0.7687    0.0291

## Veredicto: ¿v6 tiene mejor evidencia off-policy que v5?

**No; en Fusión es mas debil, y en Reduccion/emparejamiento es comparable o levemente peor.** Siendo honestos con los numeros:

1. **Fusión (el canal donde v5 tenia su unica señal significativa)**: el coeficiente de adherencia (`dist_F`) sobre el KPI refinado en
   DEV es **-0.42 pp/sd (p=0.174)** en v6 frente a **-0.70 pp/sd (p=0.025)** en v5 (tabla "Adherencia vs KPI" arriba, tomada sin
   recalcular de `e8_05_evidencia_politica_v6.csv` / `e7_05_evidencia_politica_v5.csv`). v5 alcanza significancia al 5 %; v6 no. La
   dosis-respuesta de esta iteracion (B) confirma el mismo patron para v6: `dist_F` vs `recuperacion_refinada_pct` en DEV da
   -0.398 pp/sd pero **p=0.154**, no significativo.
2. **Reduccion**: ningun sistema muestra señal de adherencia sobre el KPI (`dist_R` coef cercano a 0 y con signo adverso en ambos,
   v6 +0.063 p=0.77 DEV, v5 +0.019 p=0.93 DEV). El emparejamiento por estado (A) tampoco favorece a v6 en Reduccion: `dist_R` da
   +0.723 pp (p=0.217) en DEV pero se invierte a **-2.821 pp (p=0.054, adverso y casi significativo)** en LOCKBOX -- la señal no es
   estable entre subconjuntos, y el signo en LOCKBOX es opuesto al que apoyaria la politica.
3. **Emparejamiento agregado (A, dist_total)**: en DEV el efecto es **significativamente ADVERSO para v6** (-1.204 pp, p=0.035): los
   batches "cerca" de la politica v6 total tuvieron peor KPI refinado que los "lejos". v5 no tenia esta anomalia en E7-06 (dist_total
   DEV: -0.613 pp, p=0.315, no significativo pero mas cerca de cero).
4. **Polvo (f_polvo)**: es el hallazgo mas robusto y **es igual de fuerte en ambos sistemas** -- seguir mas de cerca la Fusión
   (`dist_F` bajo) se asocia con menos perdida a polvo: v6 +0.0063/sd (p=0.006, DEV), v5 +0.0085/sd (p=0.0002, DEV). Ambos con el
   mismo signo y orden de magnitud; no depende de que' estimador de masa se use.
5. **La cadena teorica (C) no cambia esta lectura**: proyecta un uplift puntual mayor para v6 (0.26-0.28 pp vs 0.21-0.22 pp de v5),
   pero es un calculo interno (PLM cross-fitted + pendiente OLS del propio sistema) que no esta corroborado por la evidencia
   empirica directa de los puntos 1-3. La ganancia de precision del target `m6_ln_feo_ret` (menos ruido de ensayo de %CaO, ver
   ITERACION_8_diseno.md) es real a nivel de escalon, pero todavia no se traduce en una señal off-policy mas fuerte a nivel de
   batch con los 362 batches disponibles.

**Conclusion**: con la evidencia off-policy disponible, v6 no supera a v5; en el terreno donde v5 tenia su mejor caso (adherencia en
Fusión) v6 es mas debil y dista de ser significativo. La unica mejora clara es teorica/de precision del target, no (todavia)
empirica en el KPI de batch. Recomendacion: no reemplazar v5 por v6 como politica de despliegue solo con esta evidencia; usar v6
como candidato en paralelo y revisar si mas datos (o una calibracion distinta de `masa_v6`) cierran la brecha de significancia en
Fusión, y resolver antes la discrepancia de signo en el carbon de Reduccion (punto D, 51.7 % de coincidencia) porque implica
recomendaciones operativas opuestas segun que' sistema se use.

## Limites

- **Todo es correlacional/off-policy**: ningun batch fue asignado a seguir realmente las recomendaciones; el emparejamiento (A) y
  los controles de estado (COV / `CONTROLES_BATCH` de E8-03) reducen pero no eliminan la confusion por condiciones de batch no
  observadas.
- **La cadena (C) encadena dos estimaciones en muestra/OOF** (la prediccion cross-fitted del PLM y la pendiente OLS de batch):
  hereda los supuestos de ambas (linealidad, sin confusion no observada a nivel de batch) y debe leerse como una proyeccion
  "si el modelo es correcto", no como evidencia nueva independiente.
- **Muestras chicas y resultados inestables entre subconjuntos**: el emparejamiento tiene entre 21 (LOCKBOX) y 99 (DEV) pares por
  celda; varios coeficientes cambian de signo entre DEV/LOCKBOX/TOTAL para la misma variable (p.ej. `dist_R` sobre
  `recuperacion_refinada_pct`: +0.72 pp en DEV, -2.82 pp en LOCKBOX), señal de que las estimaciones puntuales no son robustas a
  este tamaño de muestra (362 batches, 63 en lockbox).
- **El target v6 depende de la calibracion de `masa_v6.py`** (coeficientes efectivos `pG`/`gG` ajustados en DEV contra el balance
  global de batch): no se reaudita esa calibracion aqui, solo se usan los targets `m6_*` ya calculados en `e8_05_*`.
- **Discrepancia de signo en el carbon de Reduccion (D)**: v5 y v6 coinciden en la direccion recomendada solo el 51.7 % de las
  veces para `tasa_feed_Carbon_kg_min` en Reduccion (correlacion 0.273) -- la palanca mas relevante para la teoria de
  sobre-reduccion. Cualquier despliegue debe resolver primero cual sistema es correcto en ese punto antes de fiarse de cualquiera
  de los dos para esa palanca especifica.
- **Las distancias (`dist_F`/`dist_R`) se normalizan con el sd de DEV de cada palanca** (misma convencion que v5): heredan
  cualquier asimetria en la variabilidad historica de las acciones durante DEV.
