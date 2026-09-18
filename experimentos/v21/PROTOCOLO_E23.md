# E23 — carbón, GN y oxígeno acoplados

Objetivo: buscar una mejora verificable sobre v7/v11 sin declarar causalidad ni potencial de oxígeno medido a partir de caudales.

## Primera batería, fijada antes de ejecutar

Evaluación expansiva: primeros 100 batches para entrenamiento; bloques posteriores de 20. Comparar sobre exactamente las mismas filas y reportar DEV y antiguo lockbox por separado. El lockbox ya consultado no es confirmatorio. Ninguna transformación se calibra con datos posteriores al entrenamiento.

Endpoints: agotamiento Sn, retención Fe corregida por dross y cambio térmico. Referencias: estado sin acciones; estado + desvío de GN por orden (estructura de v11); estado + cuatro acciones conjuntas; estado + acciones e interacciones C×GN, C×O2total y GN×O2total. Comparador no lineal HGB para verificar si el supuesto lineal limita predicción. Variables de estado exclusivamente previas; receta conocida y orden.

Métricas: MAE y R² externo, diferencia emparejada de pérdida por batch con bootstrap de batches, desgloses por orden y periodo. No comparar correlación within con R² absoluto. Esta batería predictiva no identifica efectos causales ni autoriza optimización.

## Fundamento y cautelas

O2total = O2 + 0.21 aire. O2total − 2 GN es un excedente estequiométrico respecto de CH4, no pO2 del baño. min(GN,O2total/2) es un proxy límite de combustión completa, no GN efectivamente quemado ni calor medido. El carbón tiene demanda variable según CO/CO2, C fijo y reacción con óxidos: no adjudicar todo el oxígeno primero al GN como hecho físico.

Las interacciones prueban dependencia estadística del efecto marginal, sin atribuirles reparto químico medido. Posteriormente contrastar escenarios de C fijo y destino CO/CO2, rango de identificación, signos libres y políticas con soporte conjunto. Medir incertidumbre de inventarios y de conversión log→kg; la aproximación θ·Δa·inventario no es identidad exacta.

## Criterio de promoción

No sustituir el ganador por mejora de entrenamiento. Requerir mejora temporal emparejada relevante, estabilidad de endpoints químicos y térmicos, efecto conjunto identificable y evaluación independiente del valor de la política. Si sólo mejora predicción, conservarlo como predictor candidato. Si no mejora, registrar resultado negativo y continuar con reformulación de targets, balances y diseños de identificación.

## Adenda posterior a primera corrida

Se añadió `estado_hgb` al observar ventaja de `conjunto_hgb` sobre regresiones lineales: comparación necesaria para separar flexibilidad del estado del aporte incremental de acciones. Esta adenda es exploratoria, no preregistrada ni confirmatoria. La referencia `gn_por_orden` no es una reproducción de v11: predice niveles con estado y orden, sin intercepto de batch desconocido para un batch nuevo.

## Reproducción local

El virtualenv original conserva paquetes pero su Python base falta. Se instaló Python 3.11.15 en `.python-runtime`, sin reemplazar `.venv`. En PowerShell desde la raíz:

```powershell
$env:PYTHONPATH=Join-Path (Get-Location) '.venv/Lib/site-packages'
$env:PYTHONIOENCODING='utf-8'
& ./.python-runtime/cpython-3.11.15-windows-x86_64-none/python.exe experimentos/v21/e23_01_acoplamiento.py
```

El script sólo lee cachés locales y escribe sus propios resultados. No importa los módulos de ganadores ni altera sus configuraciones. Se excluyen filas de receta cuyos IDs no pertenecen al dataset y se exige merge uno-a-uno: la caché de receta contiene IDs vacíos repetidos ajenos a los batches de análisis.
