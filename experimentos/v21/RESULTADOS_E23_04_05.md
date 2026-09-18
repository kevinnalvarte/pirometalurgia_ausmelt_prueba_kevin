# Continuación E23: resultado final y estado no lineal

## Veredicto

No aparece un nuevo ganador defendible. La ampliación conjunta de carbón, GN, oxígeno y aire no mejora la cadena hacia resultados finales. Permitir ajuste no lineal del estado tampoco recupera una relación estable carbón–agotamiento de Sn en los dos periodos. Esto limita la identificación con estos datos; no demuestra ausencia de un mecanismo reductor del carbón.

La referencia de estructura v11 se conserva para comparación. Estos experimentos no certifican su eficacia causal ni justifican una nueva receta de planta.

## E23_04: cadena hacia resultados de batch

Entrenamiento expansivo desde 100 batches, evaluación en bloques posteriores de 20. Primera etapa: regresión con efectos absorbidos de batch y orden para retención logarítmica de Fe, ajustada por estado previo. Segunda: Ridge con contexto y score de contribución de acciones. La referencia puntúa sólo GN; los candidatos puntúan las acciones conjuntas. Se reajusta todo con datos de entrenamiento de cada bloque.

Los scores de entrenamiento de la segunda etapa son ajustados dentro de muestra en la primera etapa: queda pendiente cross-fitting también entre etapas. La evaluación externa de cada bloque no entra en ninguno de los ajustes.

**Disponibilidad temporal:** las acciones son ejecutadas y varios controles son agregados del batch. La prueba es temporal respecto del entrenamiento, pero retrospectiva respecto de las decisiones dentro del batch. No es un predictor utilizable antes de R0 ni una evaluación contrafactual de política.

MAE externo; menor es mejor:

| Resultado | Batches | Contexto | Referencia GN | Conjunto | Acoplado |
|---|---:|---:|---:|---:|---:|
| FeO equivalente neto, kg | 186 | 809.78 | 766.36 | 817.86 | 836.53 |
| Dross, pp | 256 | 3.272 | 3.172 | 3.216 | 3.235 |
| Polvo, pp | 256 | 2.808 | 2.847 | 2.846 | 2.838 |
| Recuperación refinada, pp | 256 | 3.928 | 3.885 | 3.896 | 3.914 |

En Fe, conjunto empeora MAE frente a GN en 51.50 kg, intervalo exploratorio [11.23,95.72]; acoplado empeora en 70.17 kg [16.35,130.45]. Bootstrap circular por bloques de 10 batches, 3000 réplicas, sin corrección de multiplicidad. Las restantes diferencias frente a GN no excluyen cero.

R² de referencia GN: Fe 0.005, dross −0.065, polvo −0.131 y recuperación −0.153. El desempeño absoluto es débil: el R² negativo indica más error cuadrático que usar la media del conjunto evaluado; esa media es una referencia descriptiva, no una predicción disponible de antemano. Mejor desempeño relativo no equivale a buena capacidad de recomendar.

El endpoint Fe exige observación explícita de R3. Hay 105 batches sin ese inventario final en el dataset completo; no se reemplaza R3 por R2. Se usa FeO equivalente inferido del balance, no pesaje independiente. Score lineal en inventario previo: aproximación común, no conversión log→kg exacta.

Auditoría ejecutada: 3816 predicciones, todas posteriores al corte de entrenamiento; sin duplicados batch/target/modelo; cuatro modelos sobre cada par batch/target.

## E23_05: ¿el ajuste lineal del estado ocultaba el efecto del carbón?

Se prueba una regresión parcialmente lineal con ajuste HGB del estado previo para cada target y cada acción. Cinco particiones por batch: ninguna fila comparte batch entre ajuste de funciones auxiliares y cálculo de residuos. Los coeficientes finales tienen signos libres y errores agrupados por batch. El estado incluye las cinco variables previas de E23, cuatro caudales de receta y orden; no es una reproducción del estado completo de v7.

Se estiman DEV y antiguo LB separadamente: 1084 escalones/298 batches y 242 escalones/63 batches. Esto es un diagnóstico de estabilidad, **no** validación cronológica: el cross-fitting por grupos puede usar fechas posteriores dentro del mismo periodo. El antiguo LB ya ha sido consultado y no es confirmatorio. El método no absorbe interceptos de batch y no sustituye al estimador within; comparar ambos informa sensibilidad al ajuste, no selecciona causalidad.

Tres especificaciones, todas con GN por orden, desviación del excedente estequiométrico y aire:

1. Desviación de carbón simple.
2. Desviación de carbón multiplicada por inventario Sn previo/1000.
3. Carbón simple, carbón×inventario Sn y carbón×excedente estequiométrico.

Excedente = O2 + 0.21 aire − 2 GN. La fórmula presupone GN equivalente a CH4 y caudales volumétricos en la misma base. No mide el potencial de oxígeno ni distribuye el oxidante real entre carbón, gas y baño.

Resultados relevantes para agotamiento logarítmico de Sn:

| Término / especificación | DEV, coeficiente [IC95%] | LB, coeficiente [IC95%] |
|---|---|---|
| Carbón simple | 0.000644 [−0.002933,0.004221] | −0.008021 [−0.015209,−0.000833] |
| Carbón×Sn, especificación 2 | −0.000115 [−0.000650,0.000421] | 0.000159 [−0.001389,0.001707] |
| Carbón×excedente, especificación 3 | −0.000333 [−0.000995,0.000329] | 0.000278 [−0.002361,0.002917] |

No aparece una señal reductora estable entre periodos. El carbón simple de LB tiene q BH=0.092; no supera 0.05 tras multiplicidad. En la especificación acoplada, el coeficiente simple tiene q=0.020 en LB, pero **no es el efecto marginal completo**: representa el término a inventario y excedente cero. La derivada respecto de carbón es theta_C + theta_CSn·Sn_previo/1000 + theta_Cexceso·excedente; debe evaluarse en estados con soporte y con su covarianza conjunta. Sería incorrecto traducir ese coeficiente aislado en «el carbón perjudica».

Matrices residuales de rango completo, condición estandarizada 2.03–3.34: esta prueba no presenta singularidad numérica severa. Eso no demuestra soporte suficiente para intervenciones ni ausencia de confusión.

Los intervalos son aproximados y no incluyen toda la incertidumbre del balance ni un reajuste bootstrap de las funciones auxiliares. La corrección BH se aplica por periodo a todos los tests de acciones de las tres especificaciones y targets.

## Consecuencia para el siguiente enfoque

La evidencia favorece separar tres preguntas: predecir el estado siguiente, identificar respuesta a una intervención conjunta y evaluar una política completa. No se debe usar buena predicción del estado como prueba de buen control.

Antes de promover una política nueva se necesita: disponibilidad de variables al instante de decisión; validación secuencial con acciones propuestas en lugar de consumos futuros; conversión log→kg con inventario final basal predicho; restricciones conjuntas de operación; y observaciones independientes o una prueba controlada que reduzca confusión por respuesta del operador. Los datos de gases y calidad del carbón, si existen, permitirían contrastar hipótesis actualmente representadas sólo por proxies.

Archivos reproducibles: `e23_04_cadena_prospectiva.py`, `e23_05_estado_no_lineal.py` y CSV con prefijos respectivos. Ningún ganador operativo fue reemplazado.
