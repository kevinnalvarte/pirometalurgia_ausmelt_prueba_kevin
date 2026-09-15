# E7-06 — evidencia extendida de la politica v5 (KPI refinado)

## A) Emparejamiento por estado (cerca vs lejos de la politica; dif > 0 favorece seguir la politica)

      dist                              kpi  n_pares  dif_media_pp  ci_lo  ci_hi  p_pareado  smd_medio_post  dist_media_cerca  dist_media_lejos  subset
dist_total        recuperacion_refinada_pct       99        -0.613 -1.735  0.585      0.315           0.056             0.117             0.383     DEV
dist_total          rendimiento_proxy_batch       99        -0.739 -2.042  0.451      0.236           0.056             0.117             0.383     DEV
dist_total   recuperacion_refinada_pct_next       99         0.543 -0.681  1.670      0.366           0.056             0.117             0.383     DEV
dist_total recuperacion_refinada_pct_media2       99        -0.035 -0.821  0.794      0.936           0.056             0.117             0.383     DEV
    dist_R        recuperacion_refinada_pct       99         0.131 -0.903  1.138      0.802           0.020             0.087             0.396     DEV
    dist_R          rendimiento_proxy_batch       99         0.049 -1.024  1.169      0.929           0.020             0.087             0.396     DEV
    dist_R   recuperacion_refinada_pct_next       99         0.854 -0.229  2.040      0.181           0.020             0.087             0.396     DEV
    dist_R recuperacion_refinada_pct_media2       99         0.492 -0.336  1.347      0.257           0.020             0.087             0.396     DEV
    dist_F        recuperacion_refinada_pct       99        -0.268 -1.481  0.993      0.671           0.052             0.099             0.449     DEV
    dist_F          rendimiento_proxy_batch       99        -0.238 -1.457  0.908      0.711           0.052             0.099             0.449     DEV
    dist_F   recuperacion_refinada_pct_next       99         0.599 -0.580  1.748      0.338           0.052             0.099             0.449     DEV
    dist_F recuperacion_refinada_pct_media2       99         0.166 -0.639  1.010      0.700           0.052             0.099             0.449     DEV
dist_total        recuperacion_refinada_pct       21        -0.611 -2.367  1.158      0.512           0.139             0.239             0.480 LOCKBOX
dist_total          rendimiento_proxy_batch       21        -0.638 -2.349  1.356      0.505           0.139             0.239             0.480 LOCKBOX
dist_total   recuperacion_refinada_pct_next       21         1.381 -0.737  3.532      0.235           0.140             0.239             0.480 LOCKBOX
dist_total recuperacion_refinada_pct_media2       21         0.385 -1.066  1.900      0.613           0.140             0.239             0.480 LOCKBOX
    dist_R        recuperacion_refinada_pct       21        -0.623 -2.438  1.126      0.529           0.166             0.119             0.426 LOCKBOX
    dist_R          rendimiento_proxy_batch       21        -0.295 -1.961  1.564      0.754           0.166             0.119             0.426 LOCKBOX
    dist_R   recuperacion_refinada_pct_next       21        -0.815 -2.805  1.220      0.470           0.166             0.119             0.426 LOCKBOX
    dist_R recuperacion_refinada_pct_media2       21        -0.458 -2.028  1.236      0.586           0.166             0.119             0.426 LOCKBOX
    dist_F        recuperacion_refinada_pct       21        -0.339 -1.960  1.394      0.707           0.084             0.287             0.623 LOCKBOX
    dist_F          rendimiento_proxy_batch       21        -0.394 -1.849  1.480      0.668           0.084             0.287             0.623 LOCKBOX
    dist_F   recuperacion_refinada_pct_next       21         1.679 -0.839  4.392      0.247           0.084             0.287             0.623 LOCKBOX
    dist_F recuperacion_refinada_pct_media2       21         0.670 -0.766  1.998      0.364           0.084             0.287             0.623 LOCKBOX
dist_total        recuperacion_refinada_pct      120         0.830 -0.195  1.721      0.096           0.051             0.125             0.413   TOTAL
dist_total          rendimiento_proxy_batch      120         0.832 -0.181  1.809      0.103           0.051             0.125             0.413   TOTAL
dist_total   recuperacion_refinada_pct_next      120         2.917  1.740  4.013      0.000           0.054             0.125             0.413   TOTAL
dist_total recuperacion_refinada_pct_media2      120         1.802  1.109  2.535      0.000           0.054             0.125             0.413   TOTAL
    dist_R        recuperacion_refinada_pct      120        -0.007 -0.797  0.776      0.987           0.021             0.091             0.404   TOTAL
    dist_R          rendimiento_proxy_batch      120        -0.064 -0.930  0.771      0.883           0.021             0.091             0.404   TOTAL
    dist_R   recuperacion_refinada_pct_next      120         0.313 -0.780  1.444      0.587           0.021             0.091             0.404   TOTAL
    dist_R recuperacion_refinada_pct_media2      120         0.153 -0.615  0.850      0.691           0.021             0.091             0.404   TOTAL
    dist_F        recuperacion_refinada_pct      120         0.073 -0.991  1.090      0.891           0.040             0.111             0.512   TOTAL
    dist_F          rendimiento_proxy_batch      120         0.101 -0.967  1.175      0.851           0.040             0.111             0.512   TOTAL
    dist_F   recuperacion_refinada_pct_next      120         3.531  2.404  4.608      0.000           0.041             0.111             0.511   TOTAL
    dist_F recuperacion_refinada_pct_media2      120         1.759  0.963  2.593      0.000           0.041             0.111             0.511   TOTAL

## B) Dosis-respuesta por terciles de adherencia

 subset       dist   n  kpi_cerca  kpi_medio  kpi_lejos  d_cerca_lejos_pp  proxy_cerca  proxy_lejos  ols_pp_por_sd  ols_p
    DEV dist_total 297     68.820     67.938     67.605             1.215       69.896       68.717         -0.229  0.369
    DEV     dist_R 297     69.241     67.601     67.522             1.719       70.307       68.629          0.167  0.439
    DEV     dist_F 297     68.514     68.743     67.106             1.408       69.607       68.122         -0.646  0.023
LOCKBOX dist_total  63     68.724     69.012     70.223            -1.499       69.722       71.098         -0.556  0.280
LOCKBOX     dist_R  63     68.852     69.727     69.379            -0.527       70.102       70.305         -0.342  0.507
LOCKBOX     dist_F  63     69.266     68.302     70.390            -1.124       70.294       71.337         -0.477  0.420
  TOTAL dist_total 360     68.758     68.038     68.196             0.562       69.798       69.279         -0.058  0.799
  TOTAL     dist_R 360     69.301     67.329     68.362             0.938       70.342       69.443          0.137  0.492
  TOTAL     dist_F 360     68.711     68.022     68.259             0.452       69.797       69.300         -0.274  0.272

## C) Cadena theta -> agregado -> KPI

pendientes_de                              kpi politica_en  n_batches  b_sn_dep_pp_por_unidad  b_sn_lo  b_sn_hi  b_feo_ret_pp_por_unidad  b_feo_lo  b_feo_hi  w_implicito  d_sn_dep_medio  d_feo_ret_medio  uplift_kpi_pp_medio  uplift_kpi_pp_mediana  uplift_ci_lo  uplift_ci_hi  uplift_kg_sn_medio  pct_batches_uplift_pos
          DEV        recuperacion_refinada_pct         DEV        299                   1.892    0.758    3.110                   12.021     8.443    17.331        6.353          -0.028            0.021                0.196                  0.140         0.129         0.297             100.479                  96.990
          DEV        recuperacion_refinada_pct     LOCKBOX         63                   1.892    0.758    3.110                   12.021     8.443    17.331        6.353          -0.052            0.031                0.279                  0.233         0.177         0.432             142.890                  98.413
          DEV        recuperacion_refinada_pct       TOTAL        362                   1.892    0.758    3.110                   12.021     8.443    17.331        6.353          -0.032            0.023                0.210                  0.160         0.138         0.320             107.860                  97.238
          DEV          rendimiento_proxy_batch         DEV        299                   1.248    0.014    2.543                   12.402     8.664    18.496        9.938          -0.028            0.021                0.222                  0.165         0.151         0.339             113.679                  97.659
          DEV          rendimiento_proxy_batch     LOCKBOX         63                   1.248    0.014    2.543                   12.402     8.664    18.496        9.938          -0.052            0.031                0.324                  0.281         0.217         0.498             166.233                  98.413
          DEV          rendimiento_proxy_batch       TOTAL        362                   1.248    0.014    2.543                   12.402     8.664    18.496        9.938          -0.032            0.023                0.240                  0.182         0.163         0.367             122.825                  97.790
          DEV recuperacion_refinada_pct_media2         DEV        299                   1.500    0.538    2.477                    8.410     5.066    13.362        5.608          -0.028            0.021                0.132                  0.092         0.059         0.231              67.798                  96.990
          DEV recuperacion_refinada_pct_media2     LOCKBOX         63                   1.500    0.538    2.477                    8.410     5.066    13.362        5.608          -0.052            0.031                0.186                  0.158         0.075         0.332              95.274                  98.413
          DEV recuperacion_refinada_pct_media2       TOTAL        362                   1.500    0.538    2.477                    8.410     5.066    13.362        5.608          -0.032            0.023                0.142                  0.105         0.062         0.248              72.579                  97.238
        TOTAL        recuperacion_refinada_pct         DEV        299                   2.323    1.315    3.264                   11.202     7.924    16.174        4.821          -0.028            0.021                0.167                  0.119         0.104         0.255              85.673                  96.656
        TOTAL        recuperacion_refinada_pct     LOCKBOX         63                   2.323    1.315    3.264                   11.202     7.924    16.174        4.821          -0.052            0.031                0.231                  0.203         0.133         0.366             118.200                  96.825
        TOTAL        recuperacion_refinada_pct       TOTAL        362                   2.323    1.315    3.264                   11.202     7.924    16.174        4.821          -0.032            0.023                0.178                  0.127         0.108         0.274              91.334                  96.685
        TOTAL          rendimiento_proxy_batch         DEV        299                   1.589    0.626    2.632                   11.481     8.268    15.789        7.225          -0.028            0.021                0.193                  0.147         0.131         0.278              99.073                  97.659
        TOTAL          rendimiento_proxy_batch     LOCKBOX         63                   1.589    0.626    2.632                   11.481     8.268    15.789        7.225          -0.052            0.031                0.278                  0.235         0.184         0.405             142.308                  98.413
        TOTAL          rendimiento_proxy_batch       TOTAL        362                   1.589    0.626    2.632                   11.481     8.268    15.789        7.225          -0.032            0.023                0.208                  0.160         0.140         0.299             106.598                  97.790
        TOTAL recuperacion_refinada_pct_media2         DEV        299                   1.598    0.770    2.544                    8.532     5.663    12.682        5.341          -0.028            0.021                0.132                  0.093         0.072         0.216              67.700                  96.990
        TOTAL recuperacion_refinada_pct_media2     LOCKBOX         63                   1.598    0.770    2.544                    8.532     5.663    12.682        5.341          -0.052            0.031                0.185                  0.160         0.090         0.311              94.624                  98.413
        TOTAL recuperacion_refinada_pct_media2       TOTAL        362                   1.598    0.770    2.544                    8.532     5.663    12.682        5.341          -0.032            0.023                0.141                  0.103         0.076         0.233              72.386                  97.238
