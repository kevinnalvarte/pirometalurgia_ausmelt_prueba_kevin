# E23_06–07: mejora térmica verificable y límites de la prescripción

## Resultado

Se encontró un candidato predictivo térmico mejor que las referencias de esta batería. No se encontró todavía una política conjunta superior a v11. La mejora térmica no implica que se haya identificado el efecto causal del carbón, GN, O2 o aire.

## Diseño ejecutado

E23_06 compara HGB con estado básico, estado ampliado, estado ampliado entrenado con los últimos 150 batches y selector basado exclusivamente en errores externos anteriores. Entrenamiento inicial de 100 batches, bloques futuros de 20. El selector revisa las últimas 100 posiciones de batch con predicciones previas; comienza con el básico. Hiperparámetros comunes: 150 iteraciones, 7 hojas, mínimo 20 observaciones por hoja, regularización L2=5, aprendizaje 0.05.

Las entradas no contienen consumos del escalón por predecir: estado previo, receta y orden. Se añaden SiO2, CaO, basicidad, razón Sn/FeO, lanza, gradiente térmico anterior, termocuplas y acumulados previos de carbón/GN/O2/aire. Los campos adicionales ausentes se procesan mediante el manejo de faltantes de HGB, aprendido en entrenamiento. Cada target usa su propia cohorte disponible, común a sus comparadores.

Los acumulados representan material ya suministrado, no carbono reactivo remanente ni calor almacenado. Lanza, composición y termocuplas son descriptores del proceso; no mediciones directas de cinética, viscosidad o potencial de oxígeno. El ensayo no aísla qué variable adicional produce la mejora: hace falta ablación para atribuirla.

Disponibilidad real pendiente: el código de ingeniería desplaza leyes, gradiente y termocuplas un escalón; excluye el consumo actual de los acumulados. Eso evita usar el escalón objetivo, pero no certifica hora de entrega de laboratorio ni versión de receta conocida entonces. Véase `CONTRATO_PREDECISION.md`.

## Temperatura: formular el cambio y conservar el estado inicial

E23_07 reutiliza las predicciones externas de cambio térmico de E23_06 y reconstruye:

`T_final_predicha = T_previa + deltaT_predicho`.

Se verificó fila a fila que el target observado satisface esa misma identidad. No se reajusta el modelo ni se usa temperatura final para construir una entrada. Se comparan nueve alternativas sobre los mismos 1037 escalones.

| Alternativa | MAE T final, °C | R² T final |
|---|---:|---:|
| Persistencia: repetir T previa | 14.063 | 0.8821 |
| HGB básico prediciendo T final | 14.246 | 0.8779 |
| HGB ampliado reciente prediciendo T final | 13.389 | 0.8945 |
| Incremental básico | 11.624 | 0.9189 |
| Incremental ampliado, todo el historial | 10.097 | 0.9387 |
| Incremental ampliado, últimos 150 batches | **9.945** | **0.9402** |
| Incremental con selector por errores anteriores | 10.181 | 0.9364 |

El incremental reciente reduce MAE aproximadamente 29.3% frente a persistencia y 14.4% frente al incremental básico. En el antiguo LB obtiene MAE 5.564 °C y R² 0.9087 frente a 7.105 °C y 0.8684 del incremental básico.

La diferencia de MAE, promediada primero por batch, entre incremental reciente e incremental básico es −1.743 °C, IC exploratorio [−2.242,−1.229]. Frente al predictor directo reciente es −3.445 [−5.586,−1.711]. En LB también favorece al incremental reciente: −1.541 [−2.138,−0.955] y −0.693 [−1.184,−0.216], respectivamente.

Intervalos mediante bootstrap circular de bloques de 10 batches, 3000 réplicas, sin ajuste de multiplicidad. Los datos y el antiguo LB ya fueron consultados: evidencia exploratoria, no confirmatoria. La reexpresión incremental se añadió después de observar E23_06. No se ha demostrado que 150 sea la ventana óptima.

El R² de cambio térmico del candidato reciente es 0.342; el R² 0.940 corresponde a temperatura absoluta y aprovecha el nivel previo conocido. No deben confundirse. La media de error tampoco limita el error máximo ni habilita una restricción térmica de seguridad sin calibrar incertidumbre y colas.

## Química: mejora menor y dependiente del target

| Target | MAE básico | MAE ampliado | Interpretación |
|---|---:|---:|---|
| Agotamiento logarítmico de Sn | 0.24885 | 0.24137 | Mejora pequeña, IC emparejado excluye cero en esta prueba exploratoria |
| Retención logarítmica Fe corregida por dross | 0.04846 | 0.04703 | Mejora pequeña, límite del IC próximo a cero |
| Ley Sn, pp | 0.59011 | 0.58167 | Diferencia no concluyente |
| Ley FeO equivalente, pp | 0.70816 | 0.71675 | No mejora |

Las leyes directas evitan definir el target con la masa inferida, aunque el estado básico todavía incluye inventarios inferidos. No son un experimento totalmente independiente del balance. Una ley es concentración, no extracción o recuperación: puede cambiar por variación de masa total. Mantener ambos grupos de targets ayuda a detectar si una mejora sólo aparece en la representación calculada.

## Enfoque recomendado para continuar

Conservar una arquitectura separada: predictor de estado, identificación de respuesta conjunta y selector de acciones con posibilidad de abstención. El módulo térmico incremental es el candidato que sí mejora de forma consistente en esta batería; el bloque químico ampliado sigue en evaluación. Ninguno identifica por sí solo el efecto de cambiar la receta.

La comparación no es un enfrentamiento directo de este predictor térmico contra el asesor v11: resuelven tareas distintas. La ganancia encontrada permite mejorar una pieza del futuro asesor, no declarar que el asesor completo ganó.

Pendientes concretos: ablación del estado ampliado; prueba con leyes sin inventarios inferidos en entradas; retrasos reales y trazabilidad de receta; intervalos térmicos calibrados cronológicamente; y evidencia de intervención para las acciones conjuntas. No se han cambiado consignas ni reemplazado el ganador operativo.

## Reproducción y comprobaciones

Ejecutar `e23_06_estado_predecision.py` y luego `e23_07_temperatura_incremental.py` con el runtime local documentado en `PROTOCOLO_E23.md`. Los CSV de predicciones, métricas, comparaciones y metadatos quedan junto a los scripts.

Comprobaciones ejecutadas: corte de entrenamiento estrictamente anterior a cada predicción; ausencia de duplicados batch/escalón/target/modelo; igualdad de la identidad térmica; nueve predictores sobre cada escalón de la comparación térmica. No se interpreta selección retrospectiva del mejor resultado como evaluación confirmatoria.
