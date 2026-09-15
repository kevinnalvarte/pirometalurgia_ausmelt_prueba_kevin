# E7-03 — Formas funcionales de las palancas y óptimos interiores (teoría vs datos)

Script `e7_03_formas.py` (Sonnet 5; corrido y corregido por Fable 5.1: la tabla θ de `evaluar_plm` trae `palanca`/`target` como
columnas). Salidas: `e7_03_cuadraticos.csv`, `e7_03_theta_por_tramo.csv`, `e7_03_selectividad_por_tramo.csv`, `e7_03_pd_curvas.csv`,
`e7_03_formas_vs_teoria.csv`, `e7_03_curva_J.csv`, `figs/e7_03_curva_J.png`, `figs/e7_03_pd_ale__*.png`.

## 1. Términos cuadráticos residualizados (test de Robinson dentro del PLM, estado curado, n_boot 300)

| Fase / target | R² OOF lineal → lineal + cuadrático | Cuadráticos significativos |
|---|---|---|
| R / agotamiento Sn | 0.760 → 0.759 | sólo **aire** (θ₂ −3.9·10⁻⁵ [−8.9·10⁻⁵, −3.7·10⁻⁶], cóncavo; vértice 117 Nm³/min, IC [88, 260], dentro de P5-P95 [109, 125]) |
| R / retención FeO | 0.203 → 0.191 | ninguno |
| F / Δln IRF | 0.281 → 0.275 | ninguno |
| F / agotamiento Sn | 0.187 → 0.174 | ninguno |
| F / ΔT | 0.498 → 0.494 | ninguno |

Ningún cuadrático mejora el OOF; el único significativo (aire) tiene vértice impreciso. **Veredicto: forma lineal por palanca** (con
signo de teoría, `ModeloPLMSignos`); los óptimos interiores del objetivo vienen del trade-off Sn/FeO ponderado por w, de la ventana
térmica y del soporte, no de curvaturas impuestas.

## 2. θ por tramo de avance (PLM por tramo, estado curado, palancas dosis/sobrante) — `e7_03_selectividad_por_tramo.csv`

| Tramo de avance | θ carbón → agotamiento Sn (sd/sd) | θ carbón → retención FeO | selectividad = −θ_FeO/θ_Sn | Lectura |
|---|---|---|---|---|
| < 0.84 (n 229) | +0.087 [−0.02, 0.20] | −0.0085 [−0.029, 0.000] | **0.10** | carbón selectivo: agota Sn, casi no metaliza Fe |
| 0.84-0.96 (n 179) | +0.036 | −0.012 | **0.34** | selectividad cae |
| 0.96-0.975 (n 256) | +0.028 | +0.012 (signo imposible) | no identificable | queda poco Sn; el efecto del carbón no se separa del ruido |
| > 0.975 (n 380) | +0.003 | +0.003 | no identificable | carbón sin efecto identificable en ninguna componente |

Por orden de escalón (R0-R3): R0 selectivo (−0.07), R1 no identificado, R2-R3 débil (0.04-0.22). En el tramo < 0.84 el GN sube el
agotamiento (+0.16 [0.04, 0.30]*) y el aire lo baja (−0.10 [−0.20, −0.02]*, enfría). Los IC por tramo son anchos (n 180-380): la
tendencia coincide con exp 15 (la selectividad se degrada con el avance) pero no alcanza a cuantificar un θ negativo del carbón sobre
el FeO tardío; por eso v5 modela el carbón con `Cx_sn` (efecto ∝ Sn disponible: grande temprano, ≈ 0 tarde) y deja `Cx_av` en la cota 0.

## 3. PD / ALE / SHAP del HGB estado + palancas vs tabla teórica — `e7_03_formas_vs_teoria.csv`

| Variable (rol) | agotamiento Sn (R) | retención FeO (R) | Teoría | Concuerda |
|---|---|---|---|---|
| carbón (palanca) | monótona + | no monótona compleja (≈ plana) | + / − débil | sí / parcial |
| `Cx_sn` (carbón × Sn disponible) | monótona + | plana | + / no cubierta | sí |
| dosis C/Sn disponible | óptimo interior ≈ 0.28 kg/kg (saturante; estequiométrico 0.20) | compleja | + saturante | sí |
| GN | monótona + | monótona − | + / − | sí / sí |
| exceso de O₂ | monótona − | monótona + | − / + | sí / sí |
| aire | óptimo interior ≈ 120 Nm³/min | valle | − débil / + débil | no (curvatura leve) |
| basicidad B2 previa (estado) | óptimo interior ≈ 0.53 | monótona + | óptimo interior | sí |
| T previa (estado) | valle (≈ 1120 °C) | monótona − | + saturante / − | no / sí |
| posición de lanza previa (estado) | monótona − | monótona + | valle | no (rango histórico estrecho) |
| avance previo (estado) | monótona − | monótona − | + (log) / − | no / sí |

Fusión: carbón sobre Δln IRF monótona + (contra la teoría, magnitud mínima), GN/O₂ complejas; sobre el agotamiento de Sn en Fusión GN
y O₂ monótonas + (los gases extraen Sn) y C/Sn valle. Las palancas químicas principales (carbón, GN, exceso de O₂) tienen la forma
esperada en las dos componentes de Reducción; los desacuerdos están en estados (T previa, lanza) que entran libres en g(S).

## 4. Curva del objetivo J = ln_sn_dep + 10·ln_feo_ret con los PLM lineales (estado fijo, cuatro niveles de avance)

Con palancas lineales, el argmax cae en la frontera del rango P1-P99 (carbón en el máximo, GN en el mínimo) a todo avance; la
ganancia de J respecto de la acción histórica es pequeña (0.0-0.4 unidades). Confirma que en v5 los óptimos interiores deben venir
del soporte histórico (relativo), de la ventana térmica y de la selectividad por estado, que es lo que hace el optimizador
(`e7_05_curvas_respuesta_v5.csv`).

## 5. Recomendación para v5

Forma lineal por palanca con signos de teoría; carbón vía `Cx_sn` (selectividad decreciente con el avance por construcción) y
`Cx_av`/carbón con cota ≤ 0 sobre la retención de FeO; sin cuadráticos; estados libres en g(S). Límites: IC anchos por tramo, tres
fallos de ejecución del script corregidos (índice de la tabla θ), sección 4 con HGB sin cross-fitting (descriptivo).
