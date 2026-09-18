# E12-02 — valoración por canal de pérdida (H2)

Scripts: `e12_02_lib.py`, `e12_02_01_identidad_estabilidad.py`, `e12_02_02_esquemas.py`, `e12_02_03_multiplicidad.py`,
`e12_02_04_precision_ganancia.py`. Datos: `experimentos/v9/e11_06b_tabla_batch.csv` (361 batches útiles, 298 DEV /
63 lockbox), exposiciones `xG,xC` (Reducción), controles `e11_07_lib.C_SIN_ESPESOR`.

## 1. Identidad KPI↔canales (`e12_02_identidad.csv`)

**Identidad exacta** (max |diff| = 0, corr = 1): `recuperacion_refinada_pct = 100·f_metal·(1 − sn_perdido_escoria_frac)`,
con `f_metal = 1 − f_dross − f_polvo` (suma exacta a 1, verificado a 1e-16). Es **multiplicativa**, no aditiva: la
aproximación aditiva `100·(1−f_dross−f_polvo−sn_perdido_escoria_frac)` (la que usa H2 y ya usaba `e11_07_04`) tiene
corr 0.9987 pero diff máxima de 3.19 pp — válida como aproximación de primer orden, no como identidad.

**Varianza/autocorrelación por canal** (pp, `e12_02_varianza_autocorr.csv`): sd `f_dross` 3.73, `f_polvo` 3.22,
`sn_perdido_escoria_frac` 0.75 (canal pequeño y más ruidoso en términos relativos). Autocorrelación lag-3 confirmada
en los tres (0.33 dross, 0.43 polvo, 0.14 escoria) — coherente con `e11_04` (ciclo de pesaje/atribución de 3 batches).

## 2. Estabilidad temporal de β_canal,j (`e12_02_estabilidad_resumen.csv`, ventanas móviles 80/90/100)

| canal | palanca | % ventanas mismo signo que muestra completa | estable (≥80%) |
|---|---|---|---|
| f_dross | xG (GN) | 98.9% | **sí** |
| sn_perdido_escoria_frac | xC (carbón) | 97.7% | **sí** |
| f_polvo | xG | 46.4% | no |
| f_polvo | xC | 76.3% | no |
| f_dross | xC | 52.3% | no |
| sn_perdido_escoria_frac | xG | 69.1% | no |

Confirma la hipótesis de diseño: **GN→dross y carbón→escoria son los pares estables**; el resto (incluido cualquier
vínculo de `f_polvo`) deriva de signo. La regla prospectiva sin fuga (2 ventanas disjuntas de 80 vs β expansivo,
`e12_02_clasif_prospectiva_*.csv`) coincide con la teoría a priori en los 3 canales evaluados (dross 91%, escoria
100%, polvo 27% de los 11 pasos con evidencia "estable") — ambas clasificaciones son consistentes, se usa la de
teoría a priori (más simple y sin riesgo de fuga) como primaria en C/D.

## 3. Esquemas prospectivos A–F (`e12_02_esquemas_observado.csv`, `e12_02_esquemas_pvalores.csv`)

Pendiente HC3 / t / Spearman, permutación en bloques de 20 (1500 rep.):

| esquema | total t (p perm) | DEV t (p perm) | lockbox t (p perm) |
|---|---|---|---|
| A_directo (ref. v9) | 2.05 (0.005) | 1.77 (0.010) | 0.68 (0.154) |
| B_canales_W100 | 2.10 (0.004) | 1.79 (0.010) | 0.74 (0.147) |
| C_60 (estables expansiva + polvo W60) | 1.79 (0.015) | 1.67 (0.019) | 0.33 (0.291) |
| C_100 (polvo W100) | 1.64 (0.007) | 1.39 (0.013) | 0.71 (0.152) |
| D_estables (sin polvo) | 0.97 (0.111) | 1.83 (0.016) | **−0.80 (0.773)** |
| E_eb (contracción precisión) | 1.61 (0.015) | 2.30 (0.008) | **−0.52 (0.632)** |
| F_signo (restricción GN→dross≥0, C→escoria≤0) | 2.10 (0.005) | 1.79 (0.011) | 0.74 (0.137) |

**B ≈ A**, tal como predice la linealidad (pendiente idéntica 0.528, t levemente mayor): la reconstrucción bottom-up
desde los 3 canales con la misma ventana no aporta ni resta nada frente al KPI directo. **F es bit-a-bit igual a B**:
la restricción de signo nunca se activa (β_GN→dross y β_carbón→escoria en W=100 ya tenían el signo correcto en
todas las ventanas) — la salvaguarda es gratis pero inerte con estos datos. **C, D y E son peores**: pierden t en
total/DEV y, más grave, **D y E invierten de signo en lockbox** (t −0.80 y −0.52): dar ventana expansiva a los
canales "estables" los vuelve menos adaptativos (pierden la deriva de régimen ya documentada en `e11_07_07`), y
excluir el polvo (D) o contraerlo empíricamente (E) destruye información que, aunque ruidosa, no es puro ruido.

## 4. Corrección de multiplicidad (`e12_02_multiplicidad.csv`)

| muestra | mejor esquema | t obs | p sin corregir | p corregido (máx. de 7) |
|---|---|---|---|---|
| total | B_canales_W100 | 2.38 | 0.004 | **0.010** |
| DEV | E_eb | 2.30 | 0.008 | 0.017 |
| lockbox | B_canales_W100 | 0.83 | 0.147 | 0.338 |

El mejor esquema en total (B) sobrevive la corrección (p=0.01) pero es el mismo resultado que ya tenía A sin buscar
entre esquemas. El mejor en DEV es E_eb — que es precisamente el que falla en lockbox: **buscar el mejor esquema
entre 7 sin penalización habría llevado a adoptar el peor candidato fuera de muestra**, ejemplo de la multiplicidad
que se pedía corregir.

## 5. Precisión de β y ganancia calibrada (`e12_02_precision_beta.csv`, `e12_02_ganancia.csv`)

SE mediana de la ventana en curso, β_KPI,GN y β_KPI,C: bajo B es **mayor** que bajo A (razón SE/A: GN ×1.05, carbón
×1.12) — combinar 3 regresiones de canal (aun cuando cada una usa la misma W=100) acumula varianza en vez de
reducirla; no hay ganancia de precisión por descomponer. Ganancia esperada calibrada (bootstrap bloques de 10, 1000
rep.), desplazamiento de 1 sd del puntaje: A +0.70 pp IC95 [0.28, 1.17] (741 kg Sn tercio alto−bajo); B +0.73 pp
IC95 [0.28, 1.24] (788 kg) — prácticamente idénticas. En lockbox ambos IC incluyen 0 (A [−1.87, 1.20], B [−1.49,
1.13]): ninguna gana significación fuera de muestra.

## 6. Veredicto

**No se adopta la valoración por canal en el asesor.** Se confirma H2 solo parcialmente: los pares (GN→dross,
carbón→escoria) son en efecto los estables y con signo de teoría (útiles como *diagnóstico* de mecanismo, ya
documentado en `e11_04`/`e11_07`), pero traducir eso en pesos de ventana distintos por canal **no mejora la prueba
prospectiva ni la precisión de β**, y dos de las cinco variantes que sí intentan explotar la estabilidad (D, E)
la **empeoran** e invierten de signo en lockbox. El único esquema que iguala a la referencia es B (canales, misma
ventana), que por construcción es estadísticamente indistinguible de A — no justifica la complejidad adicional. F
(restricción de signo) es una salvaguarda barata y sin costo (nunca se activó en esta campaña) pero tampoco aporta
mejora medible; se puede llevar como cinturón de seguridad sin urgencia.

**Recomendación**: mantener el KPI compuesto directo (esquema A, W=100) como valoración del asesor. Ningún esquema
cumple el criterio de suficiencia del diseño (pendiente>0 p<0.01 en total pasa para B con corrección, pero
calibración <1 en todos, relación no negativa en lockbox no se cumple con solidez —t≈0.7–0.8, n.s.— en ningún
esquema, D/E la violan directamente). Los canales siguen siendo valiosos como evidencia de *mecanismo* (§2), no
como *motor de valoración* del prescriptor.
