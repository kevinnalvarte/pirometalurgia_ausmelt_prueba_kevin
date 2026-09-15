# search_targets — ronda 2 (2026-09-14): ¿existe un target por escalón con ρ o R² > 0.70 contra el rendimiento proxy?

Punto de partida (ronda 1, README.md): SDI (Reducción) ρ 0.32 y IRF (Fusión) ρ 0.19-0.33. El KPI entre batches es casi ruido
blanco (autocorrelación 0.12, controles + tendencia R² 0.025) y el mejor modelo de batch previo daba R² OOF 0.18.

## Preguntas de la ronda 2 (paralelas)

- **ST-08 (Fable) techo de predictibilidad y fiabilidad del KPI**: (i) fiabilidad: rendimiento proxy vs recuperación real
  (metal/Sn cargado) vs rendimiento HA; (ii) R² OOF máximo con TODA la información de batch (≈150 agregados por fase +
  contexto + heel) con HGB/Ridge/ElasticNet anidados; (iii) cota de correlación alcanzable = √(fiabilidad).
- **ST-09 (Sonnet) familia 2 de candidatos teóricos**: balances de Fe y Sn por fase (Fe metalizado en Fusión = Fe cargado como
  óxido − FeO ganado; Sn/Fe del metal implícito), matriz de escoria (NBO/T, viscosidad/liquidus proxy, Al₂O₃, IRF medio en R),
  ventana térmica con sobrecalentamiento sobre liquidus estimado, canal polvo (T gases, tiro, gas total por t de carga,
  volatilización SnO ∝ exp(−E/RT)·(1−pO₂)), y compuestos teóricos SDI + IRF + polvo. Misma batería que ST-01 (Spearman/BH/OLS/
  lockbox) + Monte Carlo de ruido.
- **ST-10 (Sonnet) compuesto supervisado con validación cruzada**: biblioteca de ~40 agregados teóricos por fase (incluye los
  36 + ganadores + familia 2 si está); Lasso / ElasticNet / HGB con CV anidada (KFold por batch, 5×, repetido) sobre
  rendimiento, f_dross, f_polvo, recuperación real; R² OOF y ρ OOF, estabilidad de coeficientes, DEV→lockbox. Define el
  "compuesto teórico" con pesos aprendidos y verifica que sea aditivo por escalón.
- **ST-11 (Sonnet) KPI alternativos y canal polvo**: ¿es el KPI de dos batches (propio + siguiente, heel) más predecible?
  ¿Qué canal (dross / polvo) es predecible desde escalones y con qué? Targets de polvo por escalón (T gas pre-BHF, tiro,
  ΔT gas, gas total/carga, T baño) y su agregación; Sn en polvo vs pellets recirculados (stock).

## Criterio

Igual que la ronda 1 (DEV para elegir, lockbox sólo reporte, BH, controles). Umbral pedido: ρ o R² > 0.70. Si el techo
(ST-08) queda por debajo, se reporta el techo con su evidencia y el mejor target alcanzable.
