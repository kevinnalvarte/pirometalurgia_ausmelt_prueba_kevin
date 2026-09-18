# E11-04 (H4): ¿existe un KPI de batch con mejor señal/ruido que `recuperacion_refinada_pct`?

Scripts: `e11_04_kpi_ruido.py`, `e11_04_canales_secundarios.py`. Datos: `L.tabla_batch()` (362 batches, 299 DEV / 63
lockbox) + `feature_engineering.construir_resumen_batch_balance_global` (M, D, P, Sn_escoria en kg) + exposiciones
`x_R_gn`, `x_R_carbon` (media z del residuo en R0-R3), `x_F_carbon`, `x_F_exo2` (v9_lib). Controles = `L.CONTROLES`
(feed Sn, ley conc., espesor, carga secundaria F, dross Fe, idx cronológico).

## 1. Anatomía del ruido (`e11_04_autocorr/periodograma/mod3/crosscorr.csv`)

- **Autocorrelación a lag 3 confirmada y más amplia que el reporte previo**: f_polvo 0.43 (p 1e-17), f_dross 0.29-0.33
  (p<1e-9), f_metal 0.29 (p 3e-8), `recuperacion_refinada_pct` 0.26-0.29, `cierre_sn_frac` 0.17-0.18 y `E_kg` (Sn en
  escoria) 0.15: la periodicidad NO es exclusiva del polvo, contamina metal, dross, escoria y el cierre del balance de
  Sn — es un patrón de **todo el sistema de pesaje/atribución**, no un artefacto aislado del polvo. Lag 6 (armónico)
  también elevado en varios canales.
- **Periodograma**: sólo `f_polvo` (y, por arrastre, el KPI y `cierre_sn_frac`) tiene un pico dominante y limpio en
  periodo ≈ 3.07 batches (10.8 % de la potencia); `f_dross` tiene autocorrelación lag-3 pero el espectro es más plano
  (pico más fuerte en periodo 3.35, sólo 3.9 % de la potencia): el polvo tiene un ciclo más "puro", el dross es más
  bien persistencia de corto plazo.
- **`idx_cronologico mod 3`**: SIN diferencias de nivel (Kruskal p 0.49-0.84 en todos los canales). El ciclo de 3 no
  está anclado a una fase fija del índice absoluto — flota — coherente con un ciclo de registro/pesaje (p. ej. cada 3
  ensayos de laboratorio) más que con un patrón de turno o día fijo.
- **Compensación entre batches vecinos: NO hay evidencia.** `f_polvo_b` vs `f_polvo_{b+1}` r −0.066 (p 0.21, n.s.);
  `M_b` vs `D_{b+1}` r −0.006 (p 0.91, n.s.). La única anti-correlación fuerte es **contemporánea** (mismo batch):
  `f_metal` vs `f_dross` r −0.72 y vs `f_polvo` r −0.59 (mecánica: son fracciones del mismo total). No hay indicio de
  que metal/dross/polvo se "reasignen" de un batch al siguiente; el patrón es autocorrelación positiva del MISMO canal
  a lag 3, simultánea en varios canales — más compatible con un ciclo de pesaje/atribución compartido que con
  redistribución bilateral.
- `cierre_sn_frac` (cierre del balance de Sn) NO alterna signo (lag-1 acf ≈ 0.01-0.04, no negativo): no hay desfase de
  atribución tipo "un batch bajo seguido de uno alto"; sí hereda la periodicidad de lag 3 (0.17-0.18), consistente con
  que el cierre está contaminado por el mismo ciclo que el polvo/dross.

## 2. KPIs candidatos (fórmulas, `e11_04_kpis_batch.csv`)

| id | fórmula | sd (DEV) | sentido |
|---|---|---|---|
| a refinado (actual) | 100·M/(M+D+P+E) | 4.53 | recuperación total, E=Sn en escoria por balance |
| b sin polvo | 100·M/(M+D+E) | 4.39 | ignora el canal más ruidoso, pero deja ciego al carbón de Fusión (→polvo) |
| c3 / c5 polvo suavizado | 100·M/(M+D+P̄₃,₅+E) | 3.97 / 3.85 | P̄ = media móvil centrada; mantiene el canal, reduce su ruido |
| d3 todo suavizado | 100·M̄₃/(M̄₃+D̄₃+P̄₃+E) | 2.90 | máxima reducción de sd, máximo riesgo de artefacto |
| e legado (media con batch siguiente) | (KPI_a,b + KPI_a,b+1)/2 | 3.38 | motivado por el hallazgo de IRF con el batch siguiente |
| f recuperación real | 100·M/Sn cargado | 5.32 | normaliza por entrada, no por salidas |
| g pérdidas controlables | 100·(1−f_dross−f_escoria), mismo denom. que (a) | 3.64 | excluye el polvo del castigo (no del cómputo) |
| h medido/cargado | 100·(M+D+P)/Sn cargado | 6.98 | evita el balance de escoria (más incierto), usa el cargado |

## 3. Estadísticas comparativas (`e11_04_kpi_stats.csv`)

sd baja no es lo mismo que más señal: d3 y e bajan la sd por construcción (autocorrelación mecánica: acf1 0.77 y 0.57
respectivamente, vs 0.12 del KPI actual) sin ganar sensibilidad real (ver §5). R² OOF (Ridge/HGB, controles +
`R_sum_m6_ln_sn_dep/feo_ret`, `F_lnirf_F0`, `R_lnirf_fin`, bloques cronológicos 5) es ≈0 o negativo para casi todos
(coherente con el techo ρ≈0.3-0.4 de hallazgos §16.6, que usaba 38 agregados; aquí sólo 4 + controles); el único con
R² OOF positivo simple es `kpi_f_real` (0.069 DEV) — normaliza por una variable de entrada correlacionada con el
estado, no es evidencia de menos ruido de acción.

Sensibilidad a la acción (t HC3, joint con controles, DEV): GN_R más fuerte en c5 (t −3.15) y c3 (−3.02) que en el
actual (−2.82); carbón_R nunca significativo en ningún candidato (máx t 1.70, c5); carbón_F nunca significativo;
**exo2_F salta de t −1.87 (actual) a t −3.06/−3.12 en c3/c5** — hallazgo relevante para E11-05, no para el KPI
principal.

## 4. Prueba anidada (demostración central, `e11_04_prueba_anidada.csv`)

Con `x_R_gn` + `x_R_carbon` (signos de teoría), CV anidada gkf/crono, DEV (n 298): el KPI actual pasa (t 2.37/2.51,
p_perm 0.006/0.0045). c3 y c5 pasan con más margen (t 2.67-3.20, p_perm ≤0.0035); d3 no pasa con margen (gkf p 0.057,
n.s. a 0.05) y muestra placebo contaminado; **e (legado) y h fallan** (t 0.31-0.94, n.s. en ambos esquemas): promediar
con el batch siguiente o normalizar por el cargado sin el balance de escoria **destruye** la señal de acción, no la
mejora — descarta e y h como KPI de evaluación de decisiones.

## 5. Artefactos: HAC y placebo (`e11_04_placebo_hac.csv`)

Al corregir por autocorrelación inducida (Newey-West, lags = ventana) el aparente salto de sensibilidad de GN_R en c5
casi desaparece frente al actual (t_HAC −2.80 vs t_HC3 del actual −2.82: la ganancia nominal era en gran parte
artefacto de ventana). El placebo (exposición de b+3 → KPI suavizado de b) es limpio para GN_R (t −0.68/−0.78, n.s.)
y para exo2_F (t −1.25/−1.02, n.s.: el hallazgo de exo2_F en c3/c5 sobrevive al placebo) pero **NO para carbón_R**
(t placebo 1.9-2.1, borderline): la "mejora" de carbón_R en los KPI suavizados no es fiable, es serialidad de la
propia acción de carbón contaminando el target suavizado. d3 y e fallan el placebo en 2-3 de 4 palancas — confirma
descartarlos.

## 6. KPI secundario por canal (`e11_04_canales_secundarios.csv`)

Mecanismo directo, sin pasar por el KPI compuesto: `f_dross ~ x_R_gn` t **+3.49** (DEV, p 5e-4) vs t −2.82 del GN
dentro del KPI actual (razón t² ≈ 1.53, +53 % de potencia); `Sn_escoria_frac ~ x_R_carbon` t **−2.39** (DEV, p 0.017,
significativo — el KPI compuesto nunca llega a p<0.05 para carbón) vs t 1.34 dentro del KPI actual (razón t² ≈ 3.2,
+220 % de potencia, y cruza el umbral de significancia que el compuesto no cruza).

## 7. Veredicto (criterio 5 del diseño)

1. **No se cambia el KPI principal.** Ningún candidato lo domina simultáneamente en fiabilidad y en t de los vínculos
   aceptados sin sacrificar un canal: b/g excluyen el polvo (ciegos al carbón de Fusión, viola el requisito explícito
   de H4); e y h **pierden** la prueba anidada; d3 falla el placebo; c3/c5 ganan en la prueba anidada cruda pero la
   ganancia en GN_R se diluye con HAC y la ganancia en carbón_R no pasa el placebo — sólo el hallazgo de exo2_F es
   limpio, y es tangencial a H4.
2. **Se adoptan KPI secundarios por canal** para evidenciar mecanismos en el resto de la iteración: `f_dross` (canal
   directo de GN_R, +53 % de potencia t² y mismo signo) y `sn_perdido_escoria_frac`/Sn en escoria (canal directo de
   carbón_R, +220 % de potencia t², única vía donde carbón_R es significativo). Recomendado para E11-01/E11-03 en vez
   de o además del KPI compuesto al reportar vínculos control→resultado.
3. **KPI polvo-suavizado (ventana 3 o 5, `kpi_c3`/`kpi_c5`) se reserva como diagnóstico secundario para Fusión**
   (E11-05, canal exo2_F): usar SIEMPRE con errores HAC (Newey-West, maxlags = ventana) y validar con bloques
   cronológicos + placebo b+3; no usar su evidencia de carbón_R ni carbón_F (contaminada).
4. La periodicidad de lag 3 es un problema de **todo el sistema de pesaje/atribución** (metal, dross, polvo, cierre de
   Sn), no sólo del polvo; no hay compensación bilateral entre batches vecinos que permita corregirlo por desplazamiento.
   Sin datos de planta sobre el ciclo de pesaje no hay corrección defendible (confirma hallazgos §16.7a).
