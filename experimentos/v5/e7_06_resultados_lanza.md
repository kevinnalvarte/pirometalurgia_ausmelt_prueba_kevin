# E7-06 — evidencia extendida de la politica v5 (KPI refinado)

## A) Emparejamiento por estado (cerca vs lejos de la politica; dif > 0 favorece seguir la politica)

      dist                              kpi  n_pares  dif_media_pp  ci_lo  ci_hi  p_pareado  smd_medio_post  dist_media_cerca  dist_media_lejos  subset
dist_total        recuperacion_refinada_pct       99         0.871 -0.328  2.071      0.163           0.052             0.122             0.345     DEV
dist_total          rendimiento_proxy_batch       99         0.758 -0.475  2.007      0.245           0.052             0.122             0.345     DEV
dist_total   recuperacion_refinada_pct_next       99         0.038 -1.035  1.167      0.948           0.052             0.122             0.345     DEV
dist_total recuperacion_refinada_pct_media2       99         0.455 -0.377  1.226      0.286           0.052             0.122             0.345     DEV
    dist_R        recuperacion_refinada_pct       99         0.233 -1.061  1.474      0.714           0.024             0.114             0.454     DEV
    dist_R          rendimiento_proxy_batch       99         0.130 -1.070  1.436      0.841           0.024             0.114             0.454     DEV
    dist_R   recuperacion_refinada_pct_next       99         0.985 -0.235  2.188      0.130           0.024             0.114             0.454     DEV
    dist_R recuperacion_refinada_pct_media2       99         0.609 -0.284  1.475      0.201           0.024             0.114             0.454     DEV
    dist_F        recuperacion_refinada_pct       99        -0.658 -1.870  0.535      0.282           0.071             0.085             0.296     DEV
    dist_F          rendimiento_proxy_batch       99        -0.656 -1.819  0.568      0.300           0.071             0.085             0.296     DEV
    dist_F   recuperacion_refinada_pct_next       99         0.479 -0.570  1.528      0.386           0.071             0.085             0.296     DEV
    dist_F recuperacion_refinada_pct_media2       99        -0.090 -0.890  0.647      0.826           0.071             0.085             0.296     DEV
dist_total        recuperacion_refinada_pct       21        -1.399 -3.131  0.315      0.118           0.125             0.202             0.383 LOCKBOX
dist_total          rendimiento_proxy_batch       21        -1.135 -2.618  0.526      0.163           0.125             0.202             0.383 LOCKBOX
dist_total   recuperacion_refinada_pct_next       21         2.178  0.060  4.209      0.074           0.109             0.203             0.383 LOCKBOX
dist_total recuperacion_refinada_pct_media2       21         0.529 -0.914  1.924      0.478           0.109             0.203             0.383 LOCKBOX
    dist_R        recuperacion_refinada_pct       21        -0.361 -2.261  1.534      0.725           0.130             0.149             0.452 LOCKBOX
    dist_R          rendimiento_proxy_batch       21        -0.116 -1.801  1.794      0.907           0.130             0.149             0.452 LOCKBOX
    dist_R   recuperacion_refinada_pct_next       21        -1.363 -3.389  0.744      0.225           0.124             0.152             0.452 LOCKBOX
    dist_R recuperacion_refinada_pct_media2       21        -0.595 -2.027  0.947      0.445           0.124             0.152             0.452 LOCKBOX
    dist_F        recuperacion_refinada_pct       21         0.206 -2.038  2.732      0.871           0.053             0.199             0.397 LOCKBOX
    dist_F          rendimiento_proxy_batch       21         0.304 -1.854  2.859      0.810           0.053             0.199             0.397 LOCKBOX
    dist_F   recuperacion_refinada_pct_next       21         3.660  1.732  5.709      0.004           0.053             0.199             0.397 LOCKBOX
    dist_F recuperacion_refinada_pct_media2       21         1.933  0.606  3.252      0.015           0.053             0.199             0.397 LOCKBOX
dist_total        recuperacion_refinada_pct      120         0.010 -1.066  1.115      0.985           0.065             0.129             0.356   TOTAL
dist_total          rendimiento_proxy_batch      120         0.045 -1.054  1.270      0.938           0.065             0.129             0.356   TOTAL
dist_total   recuperacion_refinada_pct_next      120         2.312  1.201  3.457      0.000           0.065             0.129             0.356   TOTAL
dist_total recuperacion_refinada_pct_media2      120         1.161  0.391  1.883      0.005           0.065             0.129             0.356   TOTAL
    dist_R        recuperacion_refinada_pct      120        -0.040 -1.167  1.103      0.945           0.038             0.120             0.454   TOTAL
    dist_R          rendimiento_proxy_batch      120        -0.074 -1.229  1.027      0.898           0.038             0.120             0.454   TOTAL
    dist_R   recuperacion_refinada_pct_next      120         0.899 -0.230  2.143      0.122           0.039             0.121             0.454   TOTAL
    dist_R recuperacion_refinada_pct_media2      120         0.436 -0.424  1.354      0.315           0.039             0.121             0.454   TOTAL
    dist_F        recuperacion_refinada_pct      120         0.572 -0.474  1.530      0.280           0.045             0.093             0.328   TOTAL
    dist_F          rendimiento_proxy_batch      120         0.696 -0.383  1.701      0.197           0.045             0.093             0.328   TOTAL
    dist_F   recuperacion_refinada_pct_next      120         3.188  2.092  4.219      0.000           0.049             0.093             0.327   TOTAL
    dist_F recuperacion_refinada_pct_media2      120         1.817  1.005  2.564      0.000           0.049             0.093             0.327   TOTAL

## B) Dosis-respuesta por terciles de adherencia

 subset       dist   n  kpi_cerca  kpi_medio  kpi_lejos  d_cerca_lejos_pp  proxy_cerca  proxy_lejos  ols_pp_por_sd  ols_p
    DEV dist_total 297     68.734     67.696     67.933             0.801       69.741       69.021         -0.089  0.760
    DEV     dist_R 297     68.100     68.023     68.240            -0.140       69.080       69.332          0.100  0.718
    DEV     dist_F 297     68.966     67.551     67.846             1.120       70.060       68.866         -0.466  0.099
LOCKBOX dist_total  63     68.117     70.043     69.799            -1.682       69.404       70.760         -0.218  0.669
LOCKBOX     dist_R  63     69.392     68.817     69.750            -0.358       70.569       70.698          0.055  0.927
LOCKBOX     dist_F  63     68.599     69.381     69.977            -1.378       69.614       70.862         -0.590  0.200
  TOTAL dist_total 360     68.398     68.195     68.399            -0.001       69.427       69.441          0.053  0.834
  TOTAL     dist_R 360     68.328     68.061     68.604            -0.275       69.347       69.673          0.127  0.610
  TOTAL     dist_F 360     68.655     68.039     68.299             0.355       69.743       69.316         -0.137  0.561

## C) Cadena theta -> agregado -> KPI

pendientes_de                              kpi politica_en  n_batches  b_sn_dep_pp_por_unidad  b_sn_lo  b_sn_hi  b_feo_ret_pp_por_unidad  b_feo_lo  b_feo_hi  w_implicito  d_sn_dep_medio  d_feo_ret_medio  uplift_kpi_pp_medio  uplift_kpi_pp_mediana  uplift_ci_lo  uplift_ci_hi  uplift_kg_sn_medio  pct_batches_uplift_pos
          DEV        recuperacion_refinada_pct         DEV        299                   1.892    0.758    3.110                   12.021     8.443    17.331        6.353          -0.015            0.033                0.363                  0.293         0.253         0.531             186.188                  97.993
          DEV        recuperacion_refinada_pct     LOCKBOX         63                   1.892    0.758    3.110                   12.021     8.443    17.331        6.353          -0.042            0.033                0.316                  0.271         0.208         0.478             161.782                 100.000
          DEV        recuperacion_refinada_pct       TOTAL        362                   1.892    0.758    3.110                   12.021     8.443    17.331        6.353          -0.019            0.033                0.355                  0.289         0.246         0.521             181.940                  98.343
          DEV          rendimiento_proxy_batch         DEV        299                   1.248    0.014    2.543                   12.402     8.664    18.496        9.938          -0.015            0.033                0.385                  0.303         0.268         0.576             197.379                  97.993
          DEV          rendimiento_proxy_batch     LOCKBOX         63                   1.248    0.014    2.543                   12.402     8.664    18.496        9.938          -0.042            0.033                0.355                  0.294         0.243         0.543             182.067                 100.000
          DEV          rendimiento_proxy_batch       TOTAL        362                   1.248    0.014    2.543                   12.402     8.664    18.496        9.938          -0.019            0.033                0.380                  0.303         0.263         0.570             194.714                  98.343
          DEV recuperacion_refinada_pct_media2         DEV        299                   1.500    0.538    2.477                    8.410     5.066    13.362        5.608          -0.015            0.033                0.252                  0.204         0.142         0.409             128.941                  97.993
          DEV recuperacion_refinada_pct_media2     LOCKBOX         63                   1.500    0.538    2.477                    8.410     5.066    13.362        5.608          -0.042            0.033                0.213                  0.176         0.097         0.370             109.404                 100.000
          DEV recuperacion_refinada_pct_media2       TOTAL        362                   1.500    0.538    2.477                    8.410     5.066    13.362        5.608          -0.019            0.033                0.245                  0.199         0.133         0.402             125.541                  98.343
        TOTAL        recuperacion_refinada_pct         DEV        299                   2.323    1.315    3.264                   11.202     7.924    16.174        4.821          -0.015            0.033                0.330                  0.268         0.230         0.484             169.306                  97.659
        TOTAL        recuperacion_refinada_pct     LOCKBOX         63                   2.323    1.315    3.264                   11.202     7.924    16.174        4.821          -0.042            0.033                0.271                  0.215         0.170         0.410             138.714                 100.000
        TOTAL        recuperacion_refinada_pct       TOTAL        362                   2.323    1.315    3.264                   11.202     7.924    16.174        4.821          -0.019            0.033                0.320                  0.260         0.221         0.469             163.982                  98.066
        TOTAL          rendimiento_proxy_batch         DEV        299                   1.589    0.626    2.632                   11.481     8.268    15.789        7.225          -0.015            0.033                0.350                  0.280         0.248         0.485             179.472                  97.993
        TOTAL          rendimiento_proxy_batch     LOCKBOX         63                   1.589    0.626    2.632                   11.481     8.268    15.789        7.225          -0.042            0.033                0.311                  0.267         0.212         0.445             159.219                 100.000
        TOTAL          rendimiento_proxy_batch       TOTAL        362                   1.589    0.626    2.632                   11.481     8.268    15.789        7.225          -0.019            0.033                0.343                  0.277         0.243         0.479             175.947                  98.343
        TOTAL recuperacion_refinada_pct_media2         DEV        299                   1.598    0.770    2.544                    8.532     5.663    12.682        5.341          -0.015            0.033                0.254                  0.205         0.165         0.385             130.243                  97.659
        TOTAL recuperacion_refinada_pct_media2     LOCKBOX         63                   1.598    0.770    2.544                    8.532     5.663    12.682        5.341          -0.042            0.033                0.213                  0.173         0.118         0.345             109.355                 100.000
        TOTAL recuperacion_refinada_pct_media2       TOTAL        362                   1.598    0.770    2.544                    8.532     5.663    12.682        5.341          -0.019            0.033                0.247                  0.202         0.158         0.379             126.608                  98.066
