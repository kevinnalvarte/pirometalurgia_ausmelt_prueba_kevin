# E23-01: resultado de primera batería conjunta

Ejecución completada el 17/09/2026. Seis modelos × tres targets, 14 bloques temporales expansivos; predicciones externas en 962 escalones de 261 batches. Los 100 primeros batches son historia inicial; un batch posterior queda sin filas completas. Todas las comparaciones usan idénticas filas. Modelo y escaladores entrenados únicamente con posiciones anteriores al bloque.

## Hallazgo

La mejora del modelo no lineal frente al lineal procede principalmente de representar el estado, no de evidencia fuerte adicional de las acciones conjuntas. No se promueve una nueva versión ganadora.

| Target | R² HGB estado | R² HGB estado + acciones | ΔMAE medio por batch e IC95 exploratorio |
|---|---:|---:|---|
| Agotamiento Sn | 0.7756 | 0.7783 | −0.00107 [−0.00599, +0.00384] |
| Retención Fe corregida | 0.3989 | 0.4118 | −0.000828 [−0.001779, +0.000123] |
| ΔT | 0.0881 | 0.0719 | +0.124 °C [−0.108, +0.360] |

Δ negativo favorece acciones. Los tres IC incluyen cero; son bootstrap de batches, no bloques temporales, sin ajuste de multiplicidad: no deben convertirse en prueba confirmatoria de ausencia de efecto. Predicción de Fe mejora algo en promedio, pero todavía no se demuestra una mejora estable. La predicción térmica es insuficiente para gobernar restricciones de seguridad.

Interacciones lineales explícitas C×GN, C×O2total y GN×O2total apenas cambian la predicción: Sn R² 0.7531 frente a 0.7532 sin interacciones; Fe 0.3848 frente a 0.3835. No demuestran reparto químico ni potencial de oxígeno. Estos productos no representan por sí solos combustión completa/incompleta.

## Lo que se aprendió y lo que no

- El acoplamiento físico es plausible, pero agregar caudales y productos no basta para resolver identificación y calidad de estado.
- El comparador `gn_por_orden` NO es una reproducción exacta de v11: predice niveles absolutos sin efectos fijos de un batch nuevo. No interpretar esta tabla como derrota de v11.
- Se conserva la masa v6 y el target de Fe v8; por tanto, este ensayo no resuelve errores del balance.
- No hay ensayo aleatorizado, beneficio de política calculado ni recomendación nueva en este experimento.
- El HGB recibe acciones realizadas de la ventana: evaluación condicional de respuesta, no pronóstico autónomo antes de conocer las acciones ni identificación de intervenciones.
- La selección de filas exige los tres targets simultáneamente; debe complementarse con cobertura por target para evitar ocultar los R3 sin ensayo.

## Próximos experimentos necesarios para el objetivo completo

1. Residualización de acciones conjunta, colinealidad y soporte condicional por orden; estudiar si queda contraste independiente para carbón al condicionar GN/O2/aire.
2. Comparación intra-batch exacta con v11, sin usar efectos fijos futuros para fingir predicciones en nivel. Validar contribuciones y endpoints de batch con parámetros sólo del pasado.
3. Escenarios explícitos de C fijo y reparto de oxígeno; detectar si son distinguibles con estos datos o simples reparametrizaciones de los mismos caudales.
4. Propagar incertidumbre de masa, especiación y Fe del dross; usar transformación exponencial exacta para pasar cambios log a kg y distinguirla de la aproximación local de v11.
5. Sólo si hay acciones identificables, comparar políticas con soporte conjunto, incertidumbre térmica y evaluación de valor independiente. Mantener evidencia observacional separada de validación causal.

Archivos: `e23_01_acoplamiento.py`, `e23_01_predicciones.csv`, `e23_01_metricas.csv`, `e23_01_comparaciones.csv`, `e23_01_por_orden.csv`, `e23_01_metadata.json`. El hash de la caché y los conjuntos exactos están guardados en metadata. Las adendas del diseño están declaradas en `PROTOCOLO_E23.md`.
