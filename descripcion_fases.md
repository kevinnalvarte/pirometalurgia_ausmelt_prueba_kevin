# Descripción fenomenológica de las fases Fusión y Reducción — Horno Ausmelt Sn

Este documento consolida la fenomenología de las dos fases del batch (Fusión y Reducción), combinando:
(a) la evidencia empírica recalculada sobre **los 362 batches / 3982 escalones completos** del dataset
(`Datos Lingo smelter fase II.xlsx` vía `dataset_lingo_smelter.construir_dataset`), y (b) la interpretación
pirometalúrgica de esa evidencia según `consideracioness_pirometalurgia_ausmelt.md`. Todos los números de
este documento provienen de un recálculo directo sobre el dataset completo (no de un batch ilustrativo
aislado); donde una cifra proviene de un solo batch de ejemplo, se indica explícitamente.

## 1. Estructura del batch

- 362 batches, 3982 escalones: **siempre 7 escalones de Fusión + 4 de Reducción**, sin una sola excepción
  (verificado sobre el total).
- `orden_escalon_fase` indexa la posición dentro de cada fase (0-6 en Fusión, 0-3 en Reducción) y permite
  alinear escalones equivalentes entre batches distintos.
- Duración de escalón: Fusión media ≈44 min (mediana 42, rango 25-188); Reducción media ≈22 min (mediana
  25, rango 5-125) — la Reducción avanza en pasos más cortos e intensos.
- `feed_Carbon_kgh` se alimenta **en ambas fases** (no sólo en Reducción, según el diccionario de datos),
  lo cual es clave para entender la dinámica interna de Fusión (sección 2).

## 2. Fase de FUSIÓN: qué ocurre realmente (evidencia sobre los 362 batches)

### 2.1 Régimen redox

`exceso_o2_combustion_pct` (proxy de estequiometría de combustión de la lanza): mediana global **+18.9%**
(oxidante). **362/362 batches (100%)** tienen mediana de escalones de Fusión en zona oxidante — patrón
universal, no promedio que oculta excepciones.

### 2.2 La trayectoria de `ley_sn_escoria_pct` NO es monótona creciente

Contrario a la simplificación "en Fusión el %Sn de escoria sube", la trayectoria media por posición de
escalón (0→6) es:

```
16.9 → 21.5 → 21.8 (pico) → 20.9 → 19.7 → 18.4 → 16.0
```

Y el `Δ%Sn` medio por posición:

```
esc.1: +4.7   esc.2: +0.1 (techo)   esc.3: -0.7   esc.4: -1.1   esc.5: -1.2   esc.6: -2.0
```

Es decir: **Fusión son dos procesos superpuestos que compiten desde el inicio**, no un solo proceso de
acumulación:

1. **Escalones 0-2 (carga inicial):** el concentrado entra más rápido de lo que el carbón (que ya se está
   alimentando, ~13→49 kg/min en este tramo) puede reducir → acumulación neta de SnO₂ en la escoria. Patrón
   consistente en **92% de los batches** (suben de escalón 0→1).
2. **Escalones 3-6 (segunda mitad de Fusión):** el carbón acumulado ya alimentado empieza a reducir SnO₂
   tan rápido o más rápido de lo que entra nuevo concentrado → el neto se vuelve negativo y el %Sn de
   escoria **cae de forma sostenida hasta el final de Fusión**. Patrón consistente en **95% de los
   batches** (escalón 6 más bajo que el pico del escalón 2).

**Mediana global de `Δ%Sn escoria` en Fusión: -0.45** (43% de escalones positivos, 57% negativos) — la
fase, tomada en conjunto, tiene un neto ligeramente reductor, no acumulador.

### 2.3 Interpretación pirometalúrgica

- El objetivo de Fusión no es maximizar `ley_sn_escoria_pct` en cada escalón: es **construir el baño y la
  escoria** (FeO–SiO₂–CaO–Al₂O₃–MgO–SnOₓ) manteniendo preferentemente el Fe oxidado, bajo una atmósfera
  oxidante controlada que asegura combustión completa del gas natural sin reducir agresivamente el Fe
  mientras el inventario todavía se está formando (`consideracioness_pirometalurgia_ausmelt.md` §9.5).
- Que el carbón ya compita contra la acumulación desde temprano (no sólo al final) es coherente con que
  `tasa_feed_Carbon_kg_min` correlacione (Spearman) **-0.33 con `Δ%Sn escoria` incluso dentro de Fusión**
  (más débil que en Reducción, pero no nulo).
- La basicidad (`basicidad_B2 = %CaO/%SiO2`) tiene efecto marginal fuerte y monótono decreciente sobre el
  cambio de ley en Fusión (SHAP, tercil bajo→alto: +0.37→-0.43) — coherente con que aquí se está decidiendo
  la estructura de la red silicatada (polimerización SiO₂ vs. despolimerización por CaO) que condicionará
  cuánto SnO₂ queda atrapable/liberable más adelante.

## 3. Transición FUSIÓN → REDUCCIÓN: el evento más fuerte y más universal del batch

- Al cerrar la alimentación de concentrado, la misma dosis de carbón deja de "competir contra Sn entrante"
  y pasa a atacar directamente el inventario de SnOₓ ya acumulado.
- La tasa de carbón salta a su **máximo de todo el batch** justo en el primer escalón de Reducción
  (mediana ≈62 kg/min, vs. ≈45 kg/min en el último escalón de Fusión).
- Coincidiendo con ese salto, el `Δ%Sn escoria` del primer escalón de Reducción es, con diferencia, la
  mayor caída de todo el batch: **mediana ≈-12 puntos porcentuales**, presente en **361/361 batches
  evaluables (~100%)**. Este es el hallazgo más universal y más fuerte de todo el análisis.
- Conceptualmente: las mismas variables (carbón, O₂, GN) no significan lo mismo a ambos lados de la
  transición — cambia la demanda térmica, la demanda de O₂ y el objetivo metalúrgico
  (`P(y_{t+1}\mid x_t,\text{fase}=F)\neq P(y_{t+1}\mid x_t,\text{fase}=R)` aunque `x_t` sea idéntico).

*(Ilustración de un batch individual, no evidencia general por sí sola: en el batch AP0001 este salto de
carbón coincide exactamente con el cambio de exceso de O₂ de +19% a -6% y con la mayor caída de %Sn del
batch — consistente con, y confirmado por, el patrón agregado de esta sección.)*

## 4. Fase de REDUCCIÓN: qué ocurre realmente (evidencia sobre los 362 batches)

### 4.1 Régimen redox

`exceso_o2_combustion_pct`: mediana global **-4.8%** (reductor, distribución más ancha que Fusión).
**300/362 batches (83%)** tienen mediana de escalones de Reducción en zona reductora — patrón general
pero no unánime; el 17% restante es candidato a revisión (posible severidad reductora real menor a la
nominal en esos batches).

### 4.2 Trayectoria de `ley_sn_escoria_pct` y rendimientos marginales decrecientes

Nivel medio por posición (0→3): `4.0 → 2.2 → 1.6 → 1.4`. `Δ%Sn` medio por posición:

```
esc.0: -12.0   esc.1: -1.8   esc.2: -0.6   esc.3: -0.4
```

La caída se atenúa escalón a escalón **en paralelo con la reducción deliberada de la dosis de carbón**
(mediana ≈62→30→11→6 kg/min) — la planta misma reduce la severidad reductora a medida que queda menos Sn
recuperable, consistente con evitar cruzar hacia la zona donde el beneficio marginal de recuperar Sn es
menor que el costo de sobre-reducir FeO. Patrón consistente en **99% de los batches** (último escalón más
bajo que el primero de Reducción).

Correlación (Spearman) `tasa_feed_Carbon_kg_min` vs. `Δ%Sn escoria`: **-0.76** (mucho más fuerte que en
Fusión, -0.33) — la química de Reducción está dominada casi enteramente por la dosis de carbón y el estado
previo de la escoria, lo cual también explica por qué el modelo de escalón en Reducción es sustancialmente
más predecible (R² OOF hasta ≈0.96 con features expandidas) que en Fusión (R² OOF ≈0.74).

### 4.3 Interpretación pirometalúrgica

- Objetivo: extraer el SnOₓ residual de la escoria vía `SnO_2+2C\rightarrow Sn+2CO_2`, casi sin entrada de
  carga fresca (`relacion_C_carga` deja de ser interpretable aquí porque el denominador ≈0).
- Esta extracción compite con `FeO_{slag}+C\rightarrow Fe+CO`: a medida que baja el SnOₓ disponible, la
  ventaja marginal de seguir reduciendo cae mientras el riesgo de reducir Fe (hardhead, dross de Fe, peor
  calidad de metal) sube — la firma operacional de sobre-reducción es `Sn↓, FeO↓↓, Fe metal/hardhead↑`.
- La interacción `tasa_feed_Carbon_kg_min × ley_sn_escoria_pct_prev` es la validación cuantitativa directa
  de una cinética `rate \propto -k[C][SnO_x]` (tercera variable más importante del modelo en ambas fases,
  signo negativo limpio: -0.73 Fusión, -0.97 Reducción).
- `basicidad_B2` pierde casi todo su efecto marginal en Reducción (SHAP por tercil ≈0.00 a -0.04) —
  coherente con que la basicidad gobierna la *formación* de la escoria (Fusión), no la *reducción* del Sn
  ya formado.
- `posicion_vertical_lanza_mm` tiene un **óptimo interior, no un efecto monótono**: SHAP positivo en los
  extremos bajo (~1000-2000mm) y alto (>4500mm) de la variable cruda, con un valle (peor desempeño) hacia
  ~3300-4200mm — el mismo tramo concentra simultáneamente la mayor caída de %Sn (deseable) y de %FeO
  (riesgo de hardhead), consistente con una región de mayor intensidad de agitación/poder reductor que
  empuja ambas reacciones a la vez. El 64.5% de su efecto SHAP proviene de la interacción con
  `tasa_gn_nm3_min` (flujo de gas), no de un efecto propio aislado. **Pendiente de confirmar con planta**
  si un valor mayor de esta variable es lanza más alta o más baja, antes de cualquier uso prescriptivo.

## 5. Tabla resumen de diferencias Fusión vs. Reducción

| Dimensión | Fusión | Reducción |
|---|---|---|
| Escalones por batch | 7 (fijo, 362/362) | 4 (fijo, 362/362) |
| Duración media/escalón | ≈44 min | ≈22 min |
| Régimen redox (`exceso_o2_combustion_pct`) | +18.9% mediana, **100%** batches oxidante | -4.8% mediana, **83%** batches reductor |
| Trayectoria `ley_sn_escoria_pct` | Sube esc.0-2 (pico), luego cae esc.3-6 | Cae en todos los escalones, con atenuación progresiva |
| `Δ%Sn` mediana global | -0.45 (mixto, 43% positivo) | -0.99 (94% negativo) |
| Mayor caída de %Sn del batch | — | Escalón 0 de Reducción, ≈-12 pts, **~100%** de batches |
| ρ Spearman carbón vs. `Δ%Sn` | -0.33 | -0.76 |
| `relacion_C_carga` interpretable | Sí (mediana ≈0.17) | No (denominador ≈0, se usa `tasa_feed_Carbon_kg_min`) |
| R² OOF del modelo de escalón (features expandidas) | ≈0.74 | ≈0.96 |
| Variable con mayor efecto marginal | `basicidad_B2`, `ley_sn_escoria_pct_prev` | `ley_sn_escoria_pct_prev`, `ratio_sn_feo_prev` |
| Rol de `basicidad_B2` | Fuerte y monótono (formación de escoria) | Casi nulo (ya no gobierna la reducción) |
| Rol de `posicion_vertical_lanza_mm` | Monótono decreciente | Óptimo interior (~3300-4200mm es el valle) |
| Objetivo metalúrgico | Construir baño/escoria, mantener Fe oxidado | Extraer SnOₓ residual, evitar sobre-reducción de FeO |

## 6. Qué es tendencia general vs. qué es ilustración de un batch

Para trazabilidad metodológica:

- **General (validado sobre los 362 batches / 3982 escalones, o mediante `GroupKFold`/`TimeSeriesSplit`
  sobre todo el dataset):** estructura fija 7+4, régimen redox por fase, la forma no monótona de la
  trayectoria de `ley_sn_escoria_pct` en ambas fases, el salto en la transición, los rendimientos
  marginales decrecientes de Reducción, todos los R²/SHAP del modelo de escalón, y el óptimo interior de
  `posicion_vertical_lanza_mm`.
- **Ilustración de un solo batch (AP0001), explícitamente etiquetada como ejemplo, no como evidencia
  general por sí sola:** la coincidencia puntual del salto de carbón con el cambio de exceso de O₂ y la
  mayor caída de %Sn en ese batch particular — usada únicamente para dar concreción visual al patrón
  agregado de la sección 3, que sí está confirmado sobre el total.

## 7. Limitación de datos que condiciona esta fenomenología

Toda esta descripción se basa en **leyes (%) de la escoria**, no en masas, porque no existe masa de
escoria por escalón en el dataset. Esto impide cerrar un balance de masa explícito
`Δ%Sn escoria → Δ(masa de Sn en escoria)` y conectarlo directamente con `sn_en_metal_crudo_batch_t` /
`rendimiento_proxy_batch` — sigue siendo el dato de mayor valor pendiente de conseguir antes de construir
una capa prescriptiva sobre esta fenomenología.
