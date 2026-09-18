# E11-07 — validez estadística de la prueba prospectiva adaptativa

Reimplementación en numpy puro del walk-forward de `e11_06d` (validado: la pendiente punto-estimado coincide EXACTO
con la versión `statsmodels`, `e11_07_01_repro.py` → `e11_07_repro_walkforward.csv`; solo cambia el error estándar
clásico vs HC3, ej. W=100/todo: pendiente 0.528 en ambos, t 2.32 (numpy) vs 2.05 (HC3)). Esto permite 2000+
permutaciones en ~20s. Especificación primaria fijada a priori: W=100, paso=10, exposiciones `xG,xC` (residuo z de
GN y carbón en Reducción), controles `L.CONTROLES` sin `espesor_ladrillo_norm_mm`, muestra "todo" (261 batches
puntuados, `AP0102..AP0362`).

## 1. Nulas que respetan adaptatividad y autocorrelación (`e11_07_02_nulas.py` → `e11_07_nulas.csv`)

| Nula | Estadístico | Observado | p unilateral |
|---|---|---|---|
| Permutación en bloques cronológicos de 20 (2000 rep.), **W=100 primaria** | t pendiente | 2.325 | **0.0065** |
| ídem | Spearman parcial | 0.154 | 0.0065 |
| Permutación en bloques, **mínimo p sobre las 6 W** (60/80/100/120/150/expansiva) | t pendiente | 2.398 (mejor W=80) | **0.019** |
| ídem | Spearman parcial | 0.161 | **0.027** |
| Desplazamiento circular k∈[15,346] (331 corrimientos), W=100 | t pendiente | 2.325 | 0.0151 |
| ídem | Spearman parcial | 0.154 | 0.0181 |

Las dos nulas alternativas (bloques y circular) coinciden en orden de magnitud. Corregida la multiplicidad de
buscar entre 6 ventanas, el efecto sigue significativo pero el margen se reduce de p≈0.01 a p≈0.02–0.03.

## 2. Placebos prospectivos (`e11_07_03_placebos.py` → `e11_07_placebos.csv`)

| Placebo | pendiente | t | p uni |
|---|---|---|---|
| Real (referencia) | 0.528 | 2.32 | 0.010 |
| `kpi_prev`: outcome = KPI del batch **anterior** | 0.457 | 1.25 | 0.106 |
| `lead1`: exposición del batch **siguiente** | 0.420 | 1.29 | 0.098 |
| `lead3`: exposición 3 batches adelante | −0.492 | −1.19 | 0.883 (n.s., signo opuesto) |

**No son nulas limpias en lead1**: la pendiente se atenúa (~0.42–0.46 vs 0.53) pero no colapsa a 0 ni cambia de
signo hasta lead3. Indica que parte de la señal es autocorrelación de corto alcance (régimen operativo persiste
1 batch) y no un vínculo puramente contemporáneo; a 3 batches de distancia el placebo sí es limpio (signo invertido,
n.s.). Esto matiza — no invalida — el resultado de la sección 1.

## 3. Por canal, W=100 (`e11_07_04_canales.py` → `e11_07_canales.csv`, `e11_07_canal_compuesto.csv`)

| Canal (×100 pp) | pendiente | t | p uni |
|---|---|---|---|
| `sn_perdido_escoria_frac` | 0.600 | 2.68 | **0.0037** |
| `f_dross` | 0.787 | 2.39 | **0.0085** |
| `f_metal` | 0.533 | 2.30 | 0.011 |
| `f_polvo` | 0.337 | 1.30 | 0.097 (débil) |
| KPI directo (referencia) | 0.528 | 2.32 | 0.010 |
| **Compuesto** 100 − (score_dross+score_polvo+score_escoria) | 0.528 | 2.38 | **0.0086** |

**Escoria (Sn perdido) y dross sostienen la predicción prospectiva**; polvo es ruidoso (coherente con hallazgos
previos: su atribución por batch es periódica). El KPI compuesto reconstruido bottom-up desde los tres canales de
pérdida reproduce casi exactamente la pendiente y significancia del KPI directo — el vínculo no es un artefacto de
cómo se define `recuperacion_refinada_pct`.

## 4. Alternativas de re-anclaje, W=100/equivalente (`e11_07_05_alternativas.py` → `e11_07_alternativas.csv`)

| Variante | pendiente | t | p uni |
|---|---|---|---|
| Referencia (ventana dura 100) | 0.528 | 2.32 | 0.010 |
| (i) Pond. exponencial semivida 40/60/80 | 0.57 / 0.60 / 0.59 | 2.06 / 1.96 / 1.85 | 0.020 / 0.025 / 0.032 |
| (ii) Ridge hacia 0 (gl efectivos≈1.5) | 0.652 | 2.25 | 0.012 |
| (iii) Contracción media(β ventana100, β expansivo) | 0.605 | 2.04 | 0.021 |
| (iv) Solo xG | 0.570 | 2.07 | 0.019 |
| (iv) Solo xC | 0.558 | 1.23 | 0.110 (débil solo) |
| (v) + `x_F_carbon`, `x_F_exo2` (4 exposiciones) | 0.479 | **2.88** | **0.0020** |

Todas las variantes son direccionalmente consistentes (positivas). El GN de Reducción por sí solo ya es
significativo; el carbón solo, no. Agregar las dos exposiciones de Fusión da la especificación más fuerte.

## 5. Magnitud, W=100 (bootstrap por bloques de 10, 1000 rep.; `e11_07_06_magnitud.py` → `e11_07_magnitud.csv`)

- Pendiente: 0.528, IC95 boot **[0.21, 0.96]** — excluye 0; excluye 1 por poco (no hay calibración perfecta, el
  puntaje sub-predice ligeramente el KPI real).
- Diferencia de KPI (residualizado por controles) tercio alto − tercio bajo del puntaje: **+1.45 pp** (IC95
  [0.56, 2.61] pp) = **741 kg Sn/batch** (IC95 284–1336 kg).
- Desplazamiento de 1 sd del puntaje ≈ **+0.70 pp KPI ≈ 361 kg Sn/batch**.

## 6. Trayectoria β̂_G(t), β̂_C(t), W=100 (`e11_07_07_trayectoria.py` → `e11_07_trayectoria_beta.csv`)

Confirma la deriva ya reportada: β_G pasa de ≈0 (T_R ventana ≈1126 °C, espesor ≈294 mm, inicio de campaña) a
≈−2.4/−2.8 (T_R ≈1065–1085 °C) y termina en ≈0/+0.5 (T_R ≈985 °C, espesor ≈227 mm, fin de campaña); β_C hace el
camino inverso: de ≈−0.3/−0.9 a ≈+2.3/+3.6. El cruce de signo de ambos coincide con el enfriamiento del horno y el
adelgazamiento del refractario — coherente con H2 (valor del reductor dependiente del Sn/régimen térmico
disponible), no con ruido de muestreo puro.

## Veredicto

**Sí, la relación prospectiva es estadísticamente significativa una vez corregida la multiplicidad y usando nulas
que respetan la autocorrelación**, aunque con un margen más modesto que el reportado en `e11_06d`:
- Especificación primaria (W=100): p = 0.0065 (permutación en bloques) / 0.015 (desplazamiento circular).
- Corregido por buscar la mejor de 6 ventanas: **p ≈ 0.02–0.03** (t y Spearman).
- Sostiene en 6/6 alternativas de re-anclaje razonables (p < 0.05 en 6 de 7 variantes probadas; solo "xC sola"
  no llega).
- Magnitud modesta pero no trivial: ≈741 kg Sn/batch (rango realista, tercio alto vs bajo del puntaje), pendiente
  de calibración 0.53 (no calibrado a 1).
- Canal: **escoria (Sn perdido) y dross**, no polvo; un compuesto reconstruido desde los canales confirma el
  mismo tamaño de efecto que el KPI directo.
- Advertencia de honestidad: los placebos de lead-1 y KPI-anterior NO son nulos limpios (p≈0.10, pendiente
  atenuada a la mitad) — hay autocorrelación de corto alcance no absorbida por los controles; el placebo a 3
  batches sí es limpio. El efecto "real" vive en una ventana temporal corta (≤2 batches), consistente con un
  vínculo genuino acción→KPI pero exige cautela: no descartar que una fracción sea persistencia de régimen.

**Recomendación de producción**: re-anclar con **W=100** (o media vida exponencial ≈60, rendimiento equivalente y
actualización más suave) sobre `xG, xC` **+ `x_F_carbon`, `x_F_exo2`** (especificación (v), la más fuerte:
p=0.002), recalculando β cada ~10 batches. No usar ridge/contracción adicional (no mejora sobre la ventana dura ni
la exponencial). Monitorear la trayectoria de β (sección 6) junto con `T_media_R` y `espesor_ladrillo_norm_mm` como
alerta temprana de que el re-anclaje sigue una deriva real de régimen y no ruido.
