# E23-02/03 — identificación conjunta y sensibilidad del balance

Ejecutado 17/09/2026. Dos experimentos nuevos completados; no se reemplaza el ganador por estas pruebas exploratorias.

## 1. Comparación física conjunta dentro del batch

Se ajustaron tres especificaciones sobre los mismos registros válidos para cada endpoint: estructura v11 (GN por orden + carbón), conjunta (añade excedente O2 respecto de GN y aire), acoplada (añade carbón×excedente, carbón×desvío GN y carbón×Sn previo). Estado: cinco variables de v11. Efectos fijos de batch y orden; errores agrupados por batch con corrección de grados de libertad absorbidos y t de Student por número de batches. Se reportan DEV, antiguo lockbox y tres tercios DEV; q-BH por periodo sobre las pruebas de esa batería. Ni los tercios ni LB son datos confirmatorios nuevos.

La transformación de coordenadas importa: `d_exceso = d_O2 + 0.21 d_aire − 2 d_GN`. El efecto del GN a `d_exceso` y aire constantes corresponde a mover O2 a razón 2·ΔGN. No debe llamarse efecto de GN aislado. Carbón se controla como covariable, no se presupone irrelevante.

### Hallazgos

- Las interacciones de carbón sobre Sn y Fe no se identifican en DEV: p nominal de C×excedente 0.54/0.55; C×GN 0.58/0.80; C×Sn 0.31/0.66. Algunas aparecen aisladamente por tercio o LB y cambian de signo: no respaldan una política estable.
- No todo se explica por colinealidad extrema: al residualizar las acciones contra estado, efectos fijos y demás regresores, queda 27.4 % de la varianza within del carbón (VIF 3.66), frente a 49–65 % para GN por orden. Puede faltar señal, estado, resolución temporal o una forma adecuada; no se puede concluir que no exista efecto físico.
- En la especificación conjunta, GN-R3 → retención Fe es −0.00749 en DEV (p 0.0051 nominal, q 0.052) pero −0.00185 en LB (p 0.65). El signo se conserva, pero no queda demostrado un efecto preciso en cada régimen. La afirmación previa de estabilidad total debe acotarse a su especificación y endpoint.
- El excedente O2 respecto de GN se asocia a ΔT −0.770 °C/(Nm³/min) en DEV (q 0.0398) y −0.734 en LB (q 0.0374). Es una asociación ajustada, no prueba de que oxidar enfríe intrínsecamente. Operación reactiva y señales agregadas siguen siendo posibles explicaciones.

## 2. Escenarios de carbón: información nueva frente a cambio de coordenadas

Se probaron algebraicamente fracciones de C fijo 0.6/0.8/1.0 y demanda límite a CO o CO2:

`déficit = O2total − 2 GN − α · C_fijo · carbón`

Para parámetros constantes, esta variable es una combinación lineal exacta de los mismos tres caudales. El rango matricial se mantiene 3 → 3 en los seis escenarios (error máximo 1.5e-14). Agregar el déficit junto a todos los caudales no aporta una medición nueva y produce dependencia algebraica. Reemplazar caudales por ese índice puede regularizar o facilitar interpretación, pero no identifica C fijo ni destino CO/CO2. Hace falta medir composición/gases o sostener supuestos contrastables; usar saturaciones tampoco convierte el proxy en medición.

## 3. Conversión log → kg: una corrección necesaria

Con inventario inicial disponible D y target y, el inventario final es D exp(y). Para un cambio de acción que produzca Δy y conserve D:

`ΔFe_final = Fe_final_base · expm1(Δy)`

v11 usa aproximadamente `Fe_previo · Δy`. No es una identidad exacta: cambia el punto de linealización y omite el factor de retención base. Con θ archivados y estados observados, sólo como sensibilidad retrospectiva (sin conmutador, sin validación de política), el cálculo exacto condicional suma 49 772 kg frente a 53 872 kg de la aproximación, 7.6 % menos. Mediana de razón exacto/aproximado: R0 0.859; R1 0.929; R2 0.940; R3 0.941.

Esto NO recalibra automáticamente el beneficio de v11: la demostración usa el inventario final observado, que no está disponible al recomendar. Para uso previo al escalón hace falta predecir el inventario base y propagar su incertidumbre. Tampoco sustituye los intervalos económicos ni modela trayectoria posterior.

## 4. Balance: reproducción y sensibilidad

Se recalculó el cierre actual y se verificó contra caché: mismo patrón NaN y error máximo menor a 1e-6 kg (valor exacto en `e23_03_verificacion.json`). Luego 6 escenarios × 3 fracciones de Fe del dross (0.3/0.5/0.6) × 2 endpoints × 2 periodos, manteniendo filas comunes entre escenarios:

- Base; Sn expresado como SnO en lugar de SnO2; 30 % Fe ferrico en Fusión y cero en Reducción; combinación de ambos; δR=0; δR=0.02.
- Cambiar δ de 0.01 a 0 eleva masa final mediana 4.06 %; δ=0.02 la reduce 3.91 %. La masa absoluta continúa dependiendo de un parámetro no medido.
- Los efectos GN sobre Fe son poco sensibles a estos escenarios: en DEV R3 θ entre −0.00758 y −0.00723 (p nominal 0.0047–0.0055); en LB R3 entre −0.00222 y −0.00138 (p 0.58–0.73). La incertidumbre entre regímenes no desaparece ajustando estos supuestos.
- Cambiar especiación constante se cancela en buena parte por el propio cierre y sus cocientes. Esa invariancia no prueba una correcta masa física. FeIII en Fusión no cambia la masa final mediana R en esta construcción, aunque puede cambiar estados/transición; la inicialización y la conservación impuestas explican la cancelación.

Los escenarios son extremos de sensibilidad elegidos para investigar, NO rangos de confianza ni composiciones de planta medidas. No se modificaron los parámetros del ganador.

## 5. Verificación y siguiente decisión

`test_e23_identificacion.py`: dos pruebas pasadas. Se compararon coeficientes Y errores agrupados del estimador absorbido con regresión explícita con dummies en panel sintético desbalanceado; también se verificó eliminación de efectos de batch/orden. La coincidencia numérica apoya la implementación del estimador, no su identificación causal en planta.

La búsqueda no está terminada. La siguiente prueba debe comparar contribuciones físicas prospectivas y política conjunta contra la regla de receta en el mismo conjunto de batches, con control de multiplicidad y evaluación independiente. Antes de habilitar carbón, se necesita un efecto suficientemente estable o un diseño de piloto que lo identifique. La arquitectura candidata debe separar predictor de estado, estimador de efectos de acciones conjuntas y capa de decisión restringida; no premiar mejoras de R² explicadas sólo por estado.

Artefactos: scripts `e23_02_identificacion.py`, `e23_03_balance_sensibilidad.py`; CSV de efectos, cobertura, contraste independiente, rango algebraico, conversión log→kg y sensibilidad de masas. Todos quedan en esta carpeta.
