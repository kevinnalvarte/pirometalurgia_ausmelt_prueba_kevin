# E11-03 (H3) — objetivo del prescriptor anclado por el canal de la acción

Scripts: `e11_03_construir.py` (theta cross-fitted por fold externo de batches + D_Sn/D_FeO por batch, LOG y KG,
variantes "identificadas"/"todas") y `e11_03_analisis.py` (pasos 3-7). Salidas: `e11_03_tabla_batch.csv`,
`e11_03_theta_full_dev_{sn,feo}.csv`, `e11_03_anclaje.csv`, `e11_03_canales.csv`, `e11_03_sobreidentificacion.csv`,
`e11_03_cv_objetivos.csv`, `e11_03_moderacion.csv`. DEV n=297-299 batches, lockbox n=63, TOTAL n=360-361.

## 1. Paso 1-2: theta cross-fitted (sin fuga) y D_Sn/D_FeO por batch

θ ajustado en TODO DEV (referencia, `e11_03_theta_full_dev_*.csv`) replica v8: Cx_sn_v6 identificado +0.179 [0.040,
0.297], GN identificado +0.056 [0.022, 0.096] sobre Sn (`m6_ln_sn_dep`); GN identificado −0.0118 [−0.0218, −0.0034]
sobre FeO (`m8_ln_feo_ret_dross50`). Carbón sobre FeO, exceso O₂ y aire: no identificados (igual que v8). Los θ
por fold externo (5, `GroupKFold` por Batch, nunca incluyen el batch que anotan) son consistentes en magnitud y
signo con el θ de referencia. D_Sn/D_FeO por batch = Σ_t θ·(a−E[a|S])_t, versión LOG (Σ de desplazamientos del
target) y KG (Δ Sn_kg = Sn_inv_t·(1−e^−d_sn); Δ FeO_kg = FeO_inv_t·(1−e^{d_feo})), cada una con variante
"identificadas" (Cx_sn+GN para Sn; sólo GN para FeO) y "todas" (+exceso O₂, aire, y para FeO también m8_exceso_C_pos,
Cx_av_v6). D_FeO_kg positivo = FeO extra reducido por la acción (se espera coeficiente ≤0 sobre KPI); D_Sn_kg
positivo = Sn extra agotado (se espera ≥0).

## 2. Paso 3: anclaje por la acción — RESULTADO NEGATIVO

`e11_03_anclaje.csv`. Ninguna combinación (LOG/KG × identificadas/todas) da K_Sn y K_FeO con el signo teórico
estable y significativo en DEV:

| versión | muestra | K_Sn (t, p) | K_FeO (t, p) | w o λ [IC boot] |
|---|---|---|---|---|
| LOG_ident | DEV | −3.12 (−0.71, 0.48) | +5.89 (0.17, 0.86) | −1.9 [−50, 80] |
| LOG_ident | LOCKBOX | −19.1 (−1.76, 0.08) | **−163 (−2.88, 0.004)** | 8.5 [−1.9, 23.7] |
| LOG_ident | TOTAL | −5.17 (−1.32, 0.19) | −28.8 (−0.95, 0.34) | 5.6 [−20, 21] |
| KG_ident | DEV | −0.0003 (−0.32, 0.75) | −0.0025 (−1.45, 0.15) | λ −8.1 [−46, 44] |
| KG_ident | TOTAL | −0.0002 (−0.18, 0.86) | −0.0005 (−0.28, 0.78) | λ −2.6 [−18, 18] |

K_Sn tiene signo **contrario** al esperado (debería ser ≥0: más Sn agotado por la acción → más KPI) en las tres
muestras y en ambas versiones, aunque nunca significativo (|t|≤1.8): es indistinguible de ruido, no un efecto
adverso confirmado. K_FeO tiene el signo correcto en DEV log (positivo) y en las dos KG (negativo, coherente con
"FeO extra reducido perjudica"), pero nunca con |t|>1.5 en DEV/TOTAL; en LOCKBOX K_FeO es grande y significativo
pero de **signo opuesto** (régimen de fin de campaña, w≈1 ya documentado). La magnitud tampoco es físicamente
disparatada (pp por tonelada de Sn/FeO, `pp_por_t_Sn/FeO` en el CSV, del orden de la cota ~2 pp cuando el IC no
cruza 0), pero el problema es que **el IC casi siempre cruza 0**. Winsorizando D_kg al 2%/98% (16 batches
recortados) el resultado no cambia cualitativamente (sigue sin significancia) → no es un artefacto de outliers.

**Conclusión del paso 3**: a diferencia de lo esperado por H3, anclar K/w o λ únicamente con la porción de los
targets que la ACCIÓN mueve (vía θ identificado) no produce un ancla estable ni significativa en DEV con n≈300
batches: el "canal de la acción" reduce demasiado la varianza utilizable (theta cross-fitted en 2 etapas) y el
ancla queda subpotenciada.

## 3. Paso 4: canales

`e11_03_canales.csv`. El canal FeO se confirma parcialmente: KG_ident DEV, D_FeO_kg → f_dross +0.0030 (p=0.039,
signo correcto: FeO extra reducido → más dross) y → sn_perdido_escoria_frac −0.0006 (p=0.008). En TOTAL LOG_ident,
D_FeO → f_polvo +44.5 (p=0.048). El canal Sn (D_Sn → f_metal↑ / sn_perdido↓) **no se confirma** en ninguna muestra
(p≥0.45 siempre, signo inconsistente). Es decir: el mecanismo GN→FeO→dross tiene apoyo direccional aunque el
paso 3 no alcance significancia (falta de potencia, no de mecanismo); el mecanismo carbón/Cx_sn→Sn→metal no tiene
ningún apoyo a nivel de canal.

## 4. Paso 5: sobre-identificación

`e11_03_sobreidentificacion.csv`. Añadir `x_R_gn`, `x_R_carbon` (residuo crudo medio R0-R3) a la regresión de
anclaje: Wald F conjunto no significativo en DEV (LOG_ident F=4.48, p=0.11; KG_ident F=3.87, p=0.14) pero
**significativo en TOTAL** (LOG_ident F=6.57, p=0.038; KG_ident F=5.34, p=0.069, borderline), con `x_R_carbon`
significativo por sí solo en TOTAL (p=0.03, coef +1.1..+1.3). **La cadena (dos targets de escoria) no es
suficiente**: el carbón tiene un efecto directo sobre el KPI que los targets de agotamiento de Sn y retención de
FeO no capturan del todo — coherente con un canal no modelado (polvo de Fusión no aplica aquí; más probable:
temperatura/cinética de Reducción tardía, ya señalado en el dictamen v7 como "recorte tardío" no explicado).

## 5. Paso 6: comparación de objetivos por CV anidada

`e11_03_cv_objetivos.csv` (DEV: GroupKFold×10 semillas; TOTAL: 6 bloques cronológicos):

| objetivo | DEV pendiente (t, p) | TOTAL crono pendiente (t, p) | signo estable |
|---|---|---|---|
| (a) log v8, K/w fijos (targets agregados) | 2.78 (5.79, <1e-4) | 2.85 (6.41, <1e-4) | sí |
| (b) log, K^act (D_sn/D_feo log, ident) | 0.49 (1.02, 0.15) | **−0.84 (−1.24, 0.89)** | no |
| (c) kg, λ^act (D_sn/D_feo kg, ident) | 0.42 (0.87, 0.19) | **−1.74 (−2.40, 0.99)** | no |
| (d) β libres 2 exposiciones (GN_R, carbón_R crudos) | 0.73 (2.21, **0.014**) | 0.65 (1.46, 0.072) | **sí** |
| (e) (c) con restricción de signo | 0.57 (1.16, 0.12) | −1.34 (−2.03, 0.98) | no |

(a) gana en t, pero es **circular para H3**: usa los targets AGREGADOS observados (la misma varianza de
estado/mineral que motivó la iteración), no el canal de la acción — no es evidencia de que seguir la política
recomendada (vs. la práctica habitual) rinda más, sólo de que agotar más Sn y retener más FeO se asocia al KPI
(casi tautológico). Entre los candidatos genuinamente anclados en la acción, (b), (c) y (e) **cambian de signo**
entre DEV y TOTAL cronológico (mismo patrón que W≈1 en lockbox de v8): no pasan el criterio 4. (d) es el único con
signo estable y la mejor pendiente-t de los anclados en la acción, pero **no alcanza el umbral pre-registrado**
(DEV p=0.014 > 0.01 exigido; TOTAL p=0.072 > 0.05 exigido).

## 6. Paso 7: ¿reproduce la moderación GN×Sn sin parámetros extra?

`e11_03_moderacion.csv`. Con el mecanismo KG (única forma que, en principio, podría moderar sin parámetros extra,
porque Δkg = Inv·(1−e^{−d}) escala con el inventario disponible) el valor implícito de GN (+1 sd, usando K_Sn/K_FeO
de DEV o el contraste físico de v8) es **negativo en las cuatro órdenes** y el punto de cruce a cero cae fuera del
rango observado de `m6_sn_inv_kg_prev` en 3 de 4 órdenes (order 0 con el contraste v8: cruce en ~29 900 kg, cerca
del máximo histórico 29 588 → 100% de los escalones históricos quedan del lado negativo). **No reproduce** la
moderación "GN bueno con Sn alto" documentada con el residuo crudo (lockbox: GN +2.8 pp/sd con Sn alto). El valor
de carbón (sólo canal Sn vía Cx_sn) es monótono pero minúsculo (−0.0001 a −0.004 pp) y de signo negativo (heredado
del K_Sn con signo contrario al esperado del paso 3) — no interpretable con confianza dado que K_Sn no es
significativo.

## 7. Veredicto (criterio 3 y honestidad)

**H3 se rechaza en la forma propuesta.** El objetivo lineal en kg (o en log) anclado exclusivamente por el canal
de la acción (θ de escalón × residuo de la acción, K/λ ajustados sobre esa porción) no predice el KPI de batches no
vistos con la robustez exigida (criterio 4: p<0.01 DEV anidada y p<0.05 crono, mismo signo en lockbox/TOTAL): con
n≈300 batches y un θ ya cross-fitted en dos etapas, la varianza que queda para anclar K/λ es insuficiente y su
signo se invierte entre regímenes exactamente como ya le pasaba a w en v8 (fin de campaña). Tampoco reproduce la
moderación de H2 (al revés: GN queda negativo casi siempre).

El único candidato anclado en la acción que sobrevive parcialmente es **(d): dos β libres sobre los residuos
crudos de GN y carbón en Reducción** (mismo signo DEV/TOTAL: GN −, carbón +; la mejor pendiente-t de los
anclados en la acción), pero **no cumple el criterio 4** (p=0.014/0.072 vs 0.01/0.05 exigidos) — es un vigía, no
un objetivo listo para prescribir. La sobre-identificación (paso 5) confirma que ni siquiera el par de targets de
escoria agota el efecto del carbón: hace falta un canal directo (probablemente térmico/cinético en R2-R3) que
esta iteración no identifica.

**Recomendación para v9**: no reemplazar el objetivo log de v8 (K=2.78, w=5.93, anclado en targets agregados) por
ninguna de las formas ancladas en la acción probadas aquí — ni siquiera (d), que sólo alcanza para un monitoreo
(vigía) de GN/carbón, no para fijar pesos del optimizador. El canal FeO→dross del mecanismo teórico sí tiene
apoyo direccional (paso 4) y debe conservarse como justificación cualitativa, pero su magnitud sobre el KPI de
batch sigue sin poder anclarse con potencia suficiente. Recomendación operativa: sólo un piloto aleatorizado (ya
señalado en el dictamen v7 y en E10-05 §6) puede resolver esto — no hay más jugo en re-anclar con los datos
observacionales actuales.
