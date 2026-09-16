# E9-01: formas del modelo de Reduccion -- resultados

Diseno: `experimentos/v7/ITERACION_9_diseno.md` (H1). Script: `experimentos/v7/e9_01_formas.py`. Datos: `v7_lib.cargar_df()`,
`base_fase(df,'Reduccion')`, DEV/lockbox = `v7_lib.split` (299/63 batches). OOF GroupKFold(5) por Batch = criterio de seleccion;
lockbox solo reporte.

## 1. Ecuaciones finales

**ref (PLM v6, referencia)**: `y = g(S) + sum_a theta_a * (a - m_a(S))` con S = estado FULL (38 cols), A = {Cx_sn_v6, GN, exceso_o2,
aire} (sn_dep) / {carbon, Cx_av_v6, GN, exceso_o2, aire} (feo_ret); g, m HGB; theta con signo de teoria acotado (`ModeloPLMSignos`).

**tasa**: mismo PLM pero target `k = m6_ln_*/duracion_plan_min`; prediccion reescalada `pred = k_pred * duracion_plan_min` para
comparar en la escala original.

**estructural** (NLS conjunto, 7 parametros: eta_C, gamma_GN, Ea, kappa0, kappa1, epsilon, delta0):

    C_eff = feed_Carbon_kg_escalon * eta_C * exp(-Ea*(1/T - 1/T_ref)) + gamma_GN * GN_nm3_escalon * max(0, -exceso_O2/100)
    kappa = kappa0 * exp(kappa1*(B2_prev - 0.5))
    phi   = Sn_inv_prev / (Sn_inv_prev + kappa*FeO_inv_prev)
    Sn_red  = min(0.98*Sn_disp, C_eff*phi / 0.2024)                    ; Sn_disp = Sn_inv_prev + feed_Sn_kgf
    ln_sn_dep_pred  = ln(Sn_disp / (Sn_disp - Sn_red))
    FeO_red = C_eff*(1-phi)/0.1672 * epsilon
    frac_perdida = clip(FeO_red/FeO_inv_prev + delta0, -0.5, 0.98)
    ln_feo_ret_pred = ln(1 - frac_perdida)

Ajuste conjunto por `scipy.optimize.least_squares` (soft_l1, residuos normalizados por sd de cada target), OOF GroupKFold(5) por
Batch (fold ESTABLE por batch, no depende del orden de filas), bootstrap cluster por batch (220 replicas) para IC95% de parametros.
T_ref = media de T (Kelvin) en DEV.

**loglineal**: `ln(max(signo*y, 1e-3)) ~ Ridge(ln(carbon+1), ln(Sn_inv_prev), 1000/T, ln(masa_prev), B2_prev, GN, exceso_O2,
dummies orden)`, RidgeCV con StandardScaler; para feo_ret se usa signo=-1 (se modela `-m6_ln_feo_ret`, casi siempre positivo) y se
niega la prediccion.

**hibrido (a)**: estructural (mismos parametros ajustados por fold) + HGB sobre el residuo `y - pred_estructural` con estado FULL
(38 cols), fit solo en train de cada fold.

**hibrido (b)**: PLM v6 con la prediccion OOF del modelo estructural anadida como feature de estado adicional (offset), estado =
FULL + {prediccion_estructural}.

## 2. Tabla comparativa (criterio = R2 OOF DEV; lockbox y estabilidad = robustez)

                forma   clave  n_parametros  r2_oof  mae_oof  r2_lockbox  mae_lockbox identificacion_carbon estable_tercios  signo_ok
              ref_PLM  sn_dep             4  0.7652   0.2443      0.7997       0.2351                  True             NaN       NaN
              ref_PLM feo_ret             5  0.3293   0.0470      0.4861       0.0418                 False             NaN       NaN
                 tasa  sn_dep             4  0.7593   0.2495      0.8012       0.2418                   NaN             NaN       NaN
                 tasa feo_ret             5  0.3222   0.0470      0.4963       0.0422                   NaN             NaN       NaN
          estructural  sn_dep             7  0.3415   0.4141      0.4576       0.4156                  True            True       NaN
          estructural feo_ret             7  0.1346   0.0531      0.2704       0.0511                  True            True       NaN
            loglineal  sn_dep            12  0.4088   0.3274      0.6762       0.2984                   NaN             NaN       NaN
            loglineal feo_ret            12  0.1351   0.0534      0.2767       0.0501                   NaN             NaN       NaN
hibrido_a_estruct_hgb  sn_dep             7  0.6345   0.2939      0.6924       0.2940                   NaN             NaN       NaN
hibrido_a_estruct_hgb feo_ret             7  0.3204   0.0473      0.4756       0.0433                   NaN             NaN       NaN
 hibrido_b_plm_offset  sn_dep             5  0.7669   0.2439      0.8020       0.2412                   NaN             NaN       NaN
 hibrido_b_plm_offset feo_ret             6  0.3212   0.0473      0.4727       0.0420                   NaN             NaN       NaN

Notas a la tabla: `identificacion_carbon` de `estructural` = True es enganosa (ver seccion 3, es un artefacto de frontera del
optimizador, no una estimacion precisa); `estable_tercios` de `estructural` = True usa un criterio debil (ver seccion 5b). El
`ref_PLM` de `feo_ret` tiene `identificacion_carbon` = False, consistente con lo ya documentado en `hallazgos.md` #19 ("carbon
sobre FeO no identificado").

## 3. Parametros del modelo estructural (DEV, IC95% bootstrap cluster por batch, n=220)

parametro      valor      ci_lo     ci_hi  excluye_0
    eta_C     5.0000     5.0000    5.0000       True
 gamma_GN     5.0000     5.0000    5.0000       True
       Ea -1686.0630 -2111.7342 -774.7812       True
   kappa0     3.4211     3.3125    3.5538       True
   kappa1     1.6336     1.1244    2.1024       True
  epsilon     0.0208     0.0182    0.0232       True
   delta0     0.0504     0.0453    0.0556       True

`eta_C` excluye 0, pero es un **artefacto de frontera**: tanto `eta_C` como `gamma_GN` saturan en el limite superior del acotamiento
(probado en [0,5], [0,20], [0,1.5] y [0,1]: cuanto mas alto el limite, mas sube el optimo -- ver diagnostico abajo -- nunca converge
a un interior). No es identificacion real del carbon, es la optimizacion empujando la escala de `C_eff` al maximo permitido porque
el termino multiplicativo `eta_C * gamma_GN` esta pobremente identificado por separado; el "excluye 0" de la tabla no debe leerse
como evidencia de que el efecto del carbon esta bien estimado en esta forma.

**Diagnostico de la degeneracion** (bounds superiores probados sobre `eta_C`/`gamma_GN`, resto de bounds igual, ajuste sobre DEV):

| limite superior (eta_C, gamma_GN) | eta_C | gamma_GN | kappa0 | R2 sn_dep | R2 feo_ret |
|---|---|---|---|---|---|
| (5, 5)      | 5.00  | 5.00  | 3.42  | 0.348 | 0.138 |
| (20, 20)    | 15.80 | 20.00 | 12.55 | 0.373 | 0.162 |
| (1.5, 2.0)  | 1.50  | 2.00  | 0.74  | 0.065 | 0.052 |
| (1.0, 1.0)  | 1.00  | 1.00  | 0.47  | -0.146 | 0.024 |

Con bounds fisicamente defendibles (eficiencia de carbon <=100-150%), el R2 se desploma (incluso negativo en sn_dep); dejando que
el optimizador escale libremente, el R2 mejora un poco pero nunca se acerca al PLM y el ajuste se vuelve fisicamente absurdo
(`eta_C`~16, `gamma_GN`=20). Es decir: el techo de esta forma funcional esta muy por debajo del PLM **independientemente** de donde
se pongan los limites de escala.

Ademas `kappa0` > 1 en todos los casos, lo que dice que el reductor preferiria ir a FeO antes que a SnO2 (afinidad relativa del
FeO mayor que la del Sn) -- lo opuesto al prior de teoria de que el Sn se reduce preferencialmente y el exceso "se derrama" hacia
el Fe solo cuando el Sn ya esta agotado (que es justamente el patron de la tabla de selectividad de la seccion 4: la derivada de
`ln_sn_dep` cae y la de `ln_feo_ret` se mantiene casi plana con el avance). El NLS conjunto, para poder explicar el agotamiento de
Sn observado con tan poco reductor "util" (`epsilon` = 2%), termina asignando una afinidad de FeO alta y compensando con escalas
de `eta_C`/`gamma_GN` absurdas: la forma funcional propuesta es demasiado rigida para separar "selectividad Sn>FeO" de "escala del
reductor disponible" con estas 4 variables de accion. Esto es un fallo de la parametrizacion, no solo de ajuste.

## 4. Selectividad estructural: d(target)/d(carbon kg/min) por tramo de avance_prev

             der_ln_sn_dep                der_ln_feo_ret               
                      mean      std count           mean      std count
tramo_avance                                                           
<0.84              0.05115  0.05460   318       -0.00122  0.00014   318
0.84-0.96          0.03292  0.02780   232       -0.00137  0.00020   232
>0.96              0.01590  0.00936   775       -0.00118  0.00043   775

## 5. hibrido (a) vs ref (PLM v6): Delta R2 OOF (bootstrap por batch, n=500)

  clave  delta_r2_media   ci_lo   ci_hi  n_batches
 sn_dep         -0.1294 -0.2065 -0.0741        290
feo_ret         -0.0083 -0.0352  0.0165        290

hibrido (a) R2 OOF: sn_dep 0.6345 (estructural mismo fold 0.3413),
feo_ret 0.3204 (estructural mismo fold 0.1401).

hibrido (b) [PLM + offset estructural] R2 OOF: sn_dep 0.7669 (ref 0.7652), feo_ret
0.3212 (ref 0.3293).

## 5b. Estabilidad temporal (R2 OOF por tercio cronologico de DEV y walk-forward)

Nota sobre la columna `estable_tercios` de la tabla #2: aqui solo verifica que el R2 OOF sea > 0 en >= 2 de 3 tercios (una condicion
debil de "no se rompe"); NO es el criterio #2 del diseno (estabilidad del **efecto de una palanca/theta** entre tercios), que
requeriria bootstrap de theta/parametros por tercio y no se calculo por presupuesto de tiempo -- limitacion declarada.

Por tercio (R2 OOF):

| forma | clave | tercio 0 | tercio 1 | tercio 2 |
|---|---|---|---|---|
| ref (PLM) | sn_dep | 0.733 | 0.778 | 0.782 |
| ref (PLM) | feo_ret | 0.402 | 0.296 | 0.284 |
| estructural | sn_dep | **0.087** | 0.486 | 0.440 |
| estructural | feo_ret | 0.098 | 0.220 | 0.103 |

El PLM es estable entre tercios (sn_dep sube ligeramente, feo_ret baja ~0.12 pero se mantiene positivo). La forma estructural es
**inestable**: en sn_dep el primer tercio cae a 0.087 (vs 0.44-0.49 en los otros dos) -- un factor 5 de diferencia -- y en feo_ret
oscila 0.10/0.22/0.10 sin patron claro. Pese a que la columna resumen la marca "estable" bajo el criterio debil (R2>0 siempre), en
un sentido sustantivo la forma estructural NO es estable en el tiempo.

Walk-forward (entrenar en batches 1..k, predecir bloque de 50 siguientes, R2):

| bloque_desde | ref sn_dep | ref feo_ret | estructural sn_dep | estructural feo_ret |
|---|---|---|---|---|
| 50  | 0.677 | 0.336 | 0.003 | 0.112 |
| 100 | 0.765 | 0.158 | 0.282 | 0.189 |
| 150 | 0.755 | 0.318 | 0.551 | 0.235 |
| 200 | 0.720 | -0.010 | 0.100 | 0.010 |
| 250 | 0.817 | 0.340 | 0.516 | 0.139 |

El PLM ronda 0.68-0.82 en sn_dep y es mas ruidoso en feo_ret (incluye un bloque negativo, -0.01, en el tramo 200-250, coherente con
que feo_ret ya era el target mas debil). La forma estructural nunca alcanza al PLM en ningun bloque y en sn_dep parte casi en cero
(0.003 con 50 batches de entrenamiento) mejorando algo con mas historia (hasta 0.55) pero sin superar en ningun bloque al PLM.

## 6. Veredicto (criterios fijados en el diseno, seccion "Criterios de decision" #1)

- **agotamiento de Sn (sn_dep)**: ref R2 OOF 0.7652 / lockbox 0.7997; estructural R2 OOF
  0.3415 / lockbox 0.4576 (Delta R2 OOF = -0.4237). **NO SE ACEPTA (no cumple ni Delta R2 >= +0.01 ni el paquete de parsimonia+identificacion+estabilidad); el PLM v6 (ref) sigue siendo la forma ganadora**
- **retencion de FeO (feo_ret)**: ref R2 OOF 0.3293 / lockbox 0.4861; estructural R2 OOF
  0.1346 / lockbox 0.2704 (Delta R2 OOF = -0.1947). **NO SE ACEPTA (no cumple ni Delta R2 >= +0.01 ni el paquete de parsimonia+identificacion+estabilidad); el PLM v6 (ref) sigue siendo la forma ganadora**
- **hibrido (a)**: no mejora de forma estable y robusta sobre ref segun el bootstrap pareado (ver tabla #5); no se adopta como
  reemplazo, queda como diagnostico de cuanta senal no lineal falta a la fisica.
- **hibrido (b)**: no mejora de forma relevante
  al anadir el offset estructural como feature de estado del PLM.

## 7. Limitaciones y advertencias honestas

- El fold usado para el offset estructural (hibrido b) es estable por Batch (`batch_folds`) pero NO es necesariamente identico al
  fold interno que usa `evaluar_plm_signos`/`ModeloPLMSignos` (GroupKFold sobre las filas de `d_dev`, que depende del orden de
  aparicion de los batches en las filas). Riesgo de fuga leve (optimismo pequeno) en hibrido (b); no afecta a ref, tasa,
  estructural ni hibrido (a), que usan folds propios y consistentes.
- El offset estructural (`e9_offset_*`) se anadio al estado del PLM sin pasar por el clasificador de nombres `es_segura_v6_o_v5`
  (no reconoce columnas sinteticas nuevas); se justifica manualmente: es una prediccion cross-fitted (out-of-fold en el fold propio
  del modelo estructural), no un dato contemporaneo del ensayo.
- El termino de temperatura (Ea) del modelo estructural comparte identificacion con eta_C via C_eff; con solo ~10-25 grados de
  variacion tipica en T_prev dentro de DEV, Ea puede quedar debilmente identificado (ver IC en la tabla de parametros).
- delta0 (perdida basal de FeO) permite valores negativos (ganancia neta de FeO, ln_feo_ret_pred > 0), consistente con que el
  target real llega hasta +0.19 (el Fe metalico puede reoxidarse parcialmente).
- Walk-forward (`e9_01_walkforward.csv`) usa bloques de 50 batches cronologicos dentro de DEV; el primer bloque de entrenamiento
  es pequeno (50 batches) por lo que su R2 es ruidoso.
