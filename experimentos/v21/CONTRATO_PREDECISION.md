# Contrato de información para un asesor secuencial

Este contrato define el siguiente candidato; no certifica una política ni sustituye la receta.

## Instante de decisión

Inicio de cada escalón de reducción. Una variable puede usarse si su resultado estaba disponible en ese instante, no sólo si su muestra pertenece al escalón anterior.

| Información | Tratamiento |
|---|---|
| Leyes de escoria anteriores | Exigir hora de entrega del laboratorio ≤ hora de decisión. La caché no certifica esa entrega. |
| Temperatura, lanza, termocuplas anteriores | Valores registrados hasta la decisión, con antigüedad y calidad. |
| Acumulados de carbón/GN/O2/aire | Sólo consumos anteriores; `cumsum - actual` en la ingeniería existente. No equivalen a reactivo remanente. |
| Receta | Versión aprobada y conocida al decidir. El Excel no aporta trazabilidad de revisiones. |
| Inventarios de escoria | Estado inferido con supuestos de balance, acompañado por sensibilidad; no masa medida. |
| Caudales del escalón que comienza | Acciones candidatas, nunca realizaciones futuras usadas como si fueran conocidas. |
| Leyes y consumos posteriores | Sólo evaluación una vez concluye el escalón. |

## Tres componentes separados

1. **Estimador de estado y predictor bajo operación histórica.** Pronostica química y temperatura con información previa. Una buena puntuación aquí no demuestra respuesta correcta a acciones nuevas.
2. **Modelo de respuesta conjunta.** Debe evaluar carbón, GN, O2 y aire conjuntamente y representar su incertidumbre. La receta incluida en un predictor observacional no se convierte automáticamente en palanca causal; puede codificar decisiones del operador según información no registrada.
3. **Selector de acción con abstención.** Sólo modifica la referencia cuando el candidato tiene soporte conjunto, respeta restricciones verificadas y dispone de evidencia de beneficio de intervención. Si falla cualquiera, devuelve revisión/receta de referencia, sin inventar ganancia.

La abstención evita extrapolar una asociación incierta; no demuestra que la receta vigente sea óptima o segura en toda circunstancia. Los límites de planta deben proceder de especificaciones operativas validadas, no de percentiles históricos interpretados como límites físicos.

## Qué falta para evaluar una política

Reproducir secuencialmente decisiones históricas exige separar propuestas de realizaciones, retardos de laboratorio, intervenciones del operador y cambios de régimen. Una simulación con el mismo modelo que optimiza la acción sólo evalúa coherencia interna. No valida beneficio real.

La evidencia actual no permite identificar una política óptima. El candidato puede pasar primero a modo sombra: registrar estado disponible, receta, propuesta, incertidumbre y motivo de abstención, sin cambiar consignas. Después, un diseño de intervención acordado con operación debe contrastar efectos y restricciones. Estos pasos quedan propuestos; no se han ejecutado en planta.
