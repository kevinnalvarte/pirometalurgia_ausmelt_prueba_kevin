# E11-02 — ¿El valor del reductor depende del Sn disponible? (H2)

Scripts: `e11_02_moderacion.py` (tarea 1, umbral), `e11_02_forma.py` (tareas 2-3), `e11_02_temperatura.py` (tarea 4),
`e11_02_anidada.py` (tarea 5), `e11_02_escalon.py` (tarea 6); 10 CSV `e11_02_*.csv` (uno por tabla citada abajo).
Moderador primario elegido a priori: `ln ley_sn_escoria_pct_prev` (la actividad a_SnO/a_FeO escala con la
**concentración**, no con el inventario absoluto en kg); sensibilidad: `ln m6_sn_inv_kg_prev`, nivel sin ln.

## 1. b_j0, b_j1 por moderador (KPI ~ controles + mean_t(z) + mean_t(z·m̃), R0-R3, todos los moderadores)

DEV, forma lineal continua (n≈298): **ningún** b_gn1 (interacción GN×Sn) es significativo individualmente (p 0.14-0.83);
pero los 5 moderadores conceptualmente ligados a "Sn disponible" (ley_sn, m6_sn_inv, ratio_sn_feo, m6_avance con signo
invertido) apuntan al MISMO signo esperado (b_gn1 > 0, o < 0 para avance que mide lo contrario), mientras que los dos
moderadores de control (ley_feo, temperatura) no muestran patrón. Carbón: b_c1 sale **negativo** en los 5 moderadores de Sn
(-0.62 a -0.83), lo opuesto de lo que predice H2 (b_c1 ≥ 0) — ningún caso significativo pero el signo es sistemático.
Tabla completa (12 combinaciones moderador×forma) en `e11_02_tabla1_moderadores.csv`.

**Canales (DEV, primario)**: GN → f_metal −0.012 (p=0.006), f_dross +0.010 (p=0.002): repite el mecanismo ya visto en
E11-01/piloto (GN no selectivo → hardhead). Las interacciones GN×Sn en los canales **no** son significativas (mejor caso
p=0.29 en f_metal); carbón×Sn en f_metal p=0.069 (único borde de significancia, signo negativo, contrario a teoría).

## 2. Umbral m* (GN) — bootstrap 1000

Con la forma lineal el umbral existe algebraicamente pero **no está identificado**: IC bootstrap de m* va de 0 a 10^5-10^12
(problema tipo Fiéller porque b_gn1 no difiere de 0), y el punto estimado cae en el percentil 84-100 de la distribución
observada — es decir, el cruce predicho por el modelo lineal está en el borde o **fuera** del rango de Sn realmente
observado en DEV. En lockbox el signo de b_gn0/b_gn1 se **invierte** (b_gn0 +1.76, b_gn1 −0.99, ambos n.s., n=63) —
no es una rampa continua que cruza cero dentro del rango; es más compatible con un salto de nivel entre regímenes
(ver §4). Detalle en `e11_02_umbral_gn.csv`.

## 3. Forma funcional: rampa vs escalón

Terciles por orden (DEV, primario): β_GN(t1 bajo)=−2.05 (p=0.041), t2=−1.43 (p=0.18), t3 (alto)=−0.50 (p=0.41) — **atenúa
monotónicamente** hacia 0 al subir el Sn disponible, pero NO cruza a positivo dentro de DEV (consistente con el umbral
fuera de rango de §2). En TOTAL (362) sí aparece un cruce: t1=−1.90(p=0.05), t2=−1.62(p=0.09), t3=**+0.61**(p=0.34, n.s.) —
el cruce sólo emerge cuando se incluye el lockbox, reforzando que es un efecto de **régimen**, no de una rampa suave.
R² apenas mejora con terciles vs lineal (DEV 0.064 vs 0.060; TOTAL 0.066 vs 0.045): la ganancia de ajuste es marginal.
Curva β_GN(m)/β_C(m) con bandas bootstrap (lineal, DEV) en `e11_02_curva_beta.csv`: la banda de β_GN nunca excluye 0 en
todo el rango observado (1-19 %Sn); la de β_C tampoco.

## 4. Orden vs Sn disponible — compiten

`corr(interacción Sn, interacción orden)` a nivel batch = **−0.91**: casi colineales (el Sn de la escoria cae monótono
con el orden del escalón dentro de cada batch). En el modelo con ambas interacciones a la vez ("ambos"), DEV: b_gn×sn=+1.43
(p=0.17), b_gn×orden=+0.97(p=0.37) — ninguna significativa por separado (inflación de varianza esperada por la
colinealidad); en TOTAL ambas se vuelven "significativas" simultáneamente (b_gn×sn=+2.64 p=0.014, b_gn×orden=+2.30
p=0.028) pero con esa colinealidad los coeficientes individuales no son fiables — es la mezcla temperatura+régimen+orden
la que se identifica, no un efecto puro de Sn. Dentro de cada orden por separado (Rt: R0-R1, Rl: R2-R3; controla el
confusor de orden): la interacción Sn persiste en signo (Rt +0.55 p=0.42; Rl +0.70 p=0.46) pero **no es significativa**
en ninguno de los dos — subpotencia (n≈297 partido en variación residual pequeña dentro de cada tramo), no refutación
limpia. Ver `e11_02_orden_vs_sn.csv`, `e11_02_orden_dentro.csv`.

## 5. Temperatura vs Sn en TOTAL (362)

`corr(m_Sn, m_temp)` = 0.13-0.17 (moderada, no graves): en el modelo conjunto TOTAL, b_gn×temp = −0.46 (p=0.19, dirección
correcta: GN pierde valor con horno más caliente) y b_gn×sn = +0.81 (p=0.08) — ambas apuntan en la dirección de teoría
pero ninguna cruza p<0.05 controlando la otra. **Solapamiento de temperatura DEV/lockbox: crítico.** Mediana DEV 1071.6 °C
vs lockbox 975.4 °C; el **0 %** de los escalones de lockbox alcanza la mediana de DEV, y sólo el 42.3 % de DEV está por
debajo del máximo del lockbox (929-1056 °C). Coeficiente de solapamiento de densidades OVL=0.327: la mayor parte de la
masa de lockbox vive en el 15-20 % más frío de la distribución de DEV. **No se puede separar limpiamente "temperatura"
de "campaña/lockbox"** — son casi el mismo eje.

## 6. Prueba anidada (criterio 4, `e11_02_prueba_anidada.csv`)

| Espec | DEV gkf×20 (p unilat / perm) | DEV crono5 | TOTAL crono6 (362) |
|---|---|---|---|
| (a) sin moderar (GN_R, C_R) | 0.0089 / 0.0060 | 0.0061 / 0.0045 | **0.0502** / 0.0570 (falla el <0.05) |
| (b) 4 términos, ln ley_sn (primario) | 0.0083 / 0.0045 | 0.0024 / 0.0030 | **0.0220** / 0.0240 (pasa) |
| (c) 4 términos + signos (b_gn1>0, b_c0>0, b_c1≥0) | 0.0194 / 0.0085 | 0.0109 / 0.0045 | 0.0474 / 0.0475 (pasa, al límite) |
| Sensibilidad: ln m6_sn_inv_kg | 0.0026 / 0.0040 | **0.0006** / 0.0015 | 0.0267 / 0.0295 |
| Sensibilidad: ley_sn nivel | 0.0144 / 0.0105 | 0.0040 / 0.0030 | 0.0386 / 0.0390 |

Moderar por Sn disponible SÍ mejora la predicción OOF del batch 362 (crono, régimen mixto): la especificación sin
moderar falla criterio 4 en TOTAL (p≈0.05), la moderada (b) lo pasa con margen (p=0.022-0.027 según moderador). La
especificación con signos impuestos (c) es más conservadora y algo más débil (pierde el <0.01 estricto de DEV-gkf:
0.019). β medio del término de interacción GN: +0.53 a +0.58 (positivo, como predice H2) en todos los esquemas y
moderadores; β medio de la interacción de carbón: **−0.69 a −0.82** (libre) — bajo la restricción de signo colapsa a 0
en las 500 réplicas de fold (siempre en el borde), confirmando que la restricción b_c1≥0 no tiene apoyo empírico: el
dato quiere ir al lado contrario.

## 7. Coherencia con el escalón (θ_GN×Sn sobre los targets, DEV n≈1084)

| target | b_GN (main) | b_GN×Sn (ln ley_sn) | b_GN×Sn (ln m6_sn_inv) |
|---|---|---|---|
| m8_ln_feo_ret_dross50 | −0.0058 (p<0.001) | **−0.0012** (p=0.55) | −0.0016 (p=0.46) |
| m6_ln_sn_dep | +0.0436 (p<0.001) | +0.0159 (p=0.15) | +0.0156 (p=0.16) |

Los efectos principales de GN sobre ambos targets replican lo ya sabido (GN retiene menos FeO = más Fe metálico;
GN agota algo de Sn). Pero la interacción con Sn disponible **no es significativa en ningún caso**, y en `feo_ret` el
signo puntual va en la dirección **contraria** al mecanismo propuesto (se esperaba que GN redujera más FeO cuando el
Sn escasea, es decir b>0; sale −0.0012, n.s.). El mecanismo fino de selectividad a_SnO/a_FeO **no se confirma** al
nivel del escalón individual — la moderación observada a nivel de batch (§1-6) no tiene un correlato limpio en el paso
físico que se propuso como su causa.

## Veredicto (criterio 2 del diseño)

Criterio 2 exige: interacción continua p<0.05 en DEV o total; mismo signo DEV/lockbox; umbral con IC dentro del rango
observado. **Ninguno de los tres se cumple de forma limpia**: interacción lineal nunca <0.05 (mejor caso p=0.142,
sensibilidad); el signo de b_gn1 **se invierte** en lockbox (aunque n.s., n=63); el umbral no está identificado (IC
hasta 10^12) y, cuando lo está aproximadamente, cae fuera o al borde del rango observado. **H2 en su forma literal
(rampa continua con cruce de signo identificable) se rechaza.**

Lo que sí sobrevive, con matices: (i) un patrón de **atenuación tipo escalón** (bajo Sn → GN claramente negativo y
p<0.05 en el tercil bajo; alto Sn → indistinguible de 0, nunca positivo dentro de DEV); (ii) moderar por Sn **mejora
la validación OOF anidada** de 362 batches lo suficiente para cruzar el umbral de criterio 4 donde la versión sin
moderar falla al límite; (iii) el signo de la interacción de GN es consistente (positivo) en 5/5 moderadores
relacionados con Sn y en los 5 esquemas de CV de la tarea 5 — hay señal real, aunque débil y no identificada
individualmente. (iv) El moderador Sn y el "orden del escalón" están casi confundidos (r=−0.91 en la interacción);
dentro de cada orden la señal persiste en signo pero pierde significancia (subpotencia, no refutación). (v) Temperatura
y régimen (DEV/lockbox) están casi confundidos (solapamiento 33 %) y compiten por la misma varianza que Sn en TOTAL.
(vi) **El mecanismo predicho para carbón (b_c1≥0) se refuta**: el dato consistentemente prefiere b_c1<0 (aunque n.s.).
(vii) El mecanismo fino a nivel de escalón (θ_GN×Sn sobre feo_ret/sn_dep) **no aparece** — la moderación de batch no
tiene respaldo físico directo confirmado con la potencia actual.

**Recomendación para la ola 2**: no incorporar un término de moderación libre y sin restricción de GN×Sn al objetivo v9
(no está identificado, riesgo de sobreajuste al régimen lockbox/temperatura); si se usa alguna forma de moderación,
preferir la versión **por tramos/orden** (Rt vs Rl) o con restricción de signo débil, y tratarla como *hipótesis con
apoyo parcial* más que como palanca confirmada. El hallazgo más sólido y accionable de este experimento es indirecto:
la ganancia de la prueba anidada TOTAL al pasar de "sin moderar" a "moderado" (0.050→0.022 de p) sugiere que ALGO del
régimen tardío de campaña (Sn y/o temperatura, confundidos) sí cambia el valor de GN, pero E11-02 no logra aislar cuál
de los dos —ni demostrar el mecanismo físico— con los datos y la potencia disponibles.
