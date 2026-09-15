# E7-04 — Parsimonia del estado, estabilidad temporal de θ y reentrenamiento por campaña

Script `e7_04_estabilidad.py` (Sonnet 5, corrido por etapas por Fable 5.1: parsimonia | periodos | walkforward | reentrenamiento |
deriva); memo consolidado por Fable. Complemento `e7_07_parsimonia_v5.py` (re-evaluación de los sets parsimoniosos con las palancas y
signos definitivos de v5). Salidas: `e7_04_parsimonia.csv`, `e7_04_set_parsimonioso.json`, `e7_04_theta_periodos*.csv`,
`e7_04_walkforward.csv`, `e7_04_reentrenamiento.csv`, `e7_04_deriva.csv`, `e7_07_parsimonia_v5.csv`.

## 1. Parsimonia del estado (eliminación hacia atrás por importancia de permutación; PLM completo en cada k)

| Target (fase) | k FULL → R² OOF / lockbox | k* (regla ≥ 0.98·máx) | R² OOF / lockbox en k* | Features retenidas en k* |
|---|---|---|---|---|
| agotamiento Sn (R) | 37 → 0.764 / 0.800 | **5** | 0.764 / 0.803 (máx 0.774 en k = 10-16) | %Sn prev, Sn×FeO prev, lanza prev, aire acumulado, Sn inventario prev |
| retención FeO (R) | 37 → 0.165 / 0.190 | **16** | **0.219 / 0.266** | %CaO, B2, B4, índice estructural, IRF, Sn/FeO, Sn×FeO, FeO×B2, grad T, carga/cal/carbón acumulados, inventarios Sn y FeO, avance, ley Sn carga |
| Δln IRF (F) | 45 → 0.277 / 0.087 | 5 | 0.415 / −0.016 | %FeO prev, IRF prev, termocupla, masa, orden |
| Sn extraído kg (F) | 45 → 0.659 / 0.473 | 5 | 0.722 / 0.642 | reciclo y carga secundaria acumulados, inventarios Sn y FeO, tiempo de fase |

Con las palancas y signos de v5 (`e7_07`): agotamiento k = 10 → R² 0.775 / 0.807, `Cx_sn` +0.117 [0, 0.229] (queda en la cota: pierde
la significancia que tiene con 37, +0.183 [0.008, 0.288]) y GN +0.064 [0.018, 0.097]*; retención de FeO k = 16 → 0.223 / 0.263, GN
colapsa a 0 [−0.011, 0] y el exceso de O₂ pasa a +0.007 [0.001, 0.012]*. **Veredicto**: la parsimonia mejora el ajuste 0.01-0.05 pero
cambia la palanca identificada (carbón/GN → GN/exceso O₂). v5 adopta el estado completo (identifica el reductor selectivo y el no
selectivo, coherente con exp 19 y ST-12) y deja los sets de 10/16 como variante de producción.

## 2. Estabilidad temporal de θ (tercios cronológicos de DEV + DEV completo + lockbox; estado curado)

| Target / palanca | signo por período (T1, T2, T3, DEV, lockbox) | significativo en k períodos | Lectura |
|---|---|---|---|
| agotamiento: `Cx_sn` | + + + + + | 2 | estable (θ_lockbox/θ_DEV 1.16) |
| agotamiento: GN | + + + + + | 0-1 | estable en signo |
| agotamiento: exceso O₂ | + − + − + | 0 | ≈ 0 |
| retención FeO: GN | + − − − + | 1 | negativo en la campaña media; positivo al inicio y en el lockbox (n = 63): frágil |
| retención FeO: carbón / C×avance | mixto | 0 | no identificado → cota 0 en v5 |
| Fusión Sn extraído: GN | + + + + + | 2 | estable |
| Fusión: C/Sn, aire, exceso O₂ | mixto | 0 | no identificados |

## 3. Walk-forward cronológico (bloques de 50 batches, DEV)

Agotamiento 0.744 global (bloques 0.66-0.83) vs 0.762 OOF aleatorio (brecha 0.02); retención de FeO 0.144 vs 0.206 (brecha 0.06;
bloques −0.06 a 0.34); SDI 0.09 vs 0.16; Fusión Δln IRF 0.23 vs 0.28; Sn extraído 0.63 vs 0.69. La brecha es pequeña en el
agotamiento y moderada en el FeO (el componente ruidoso y dependiente de campaña).

## 4. Reentrenamiento por campaña sobre el lockbox

| Target | fijo DEV | progresivo cada 10 batches | ventana móvil 150 |
|---|---|---|---|
| agotamiento Sn | 0.805 | **0.812** | 0.799 |
| retención FeO | 0.122 | **0.286** | 0.224 |
| SDI | 0.141 | **0.216** | 0.155 |
| Fusión Sn extraído | 0.618 | **0.663** | 0.631 |

θ del progresivo al final ≈ fijo (GN sobre retención −0.002/−0.003 por unidad). **Recomendación**: producción con reentrenamiento
progresivo (ventana expansiva) cada ~10 batches.

## 5. Deriva DEV → lockbox

Ninguna feature de los sets parsimoniosos está 100 % fuera de [P1, P99] (a diferencia de `espesor_ladrillo_norm_mm`, que no entra en
ellos). Mayor deriva (KS): exceso de O₂ en Reducción 0.63 (media −1.8 % → +1.8 %: lanza más oxidante al fin de campaña), GN de
Fusión 0.62, termocupla de Fusión 0.87 (horno ≈ 60 °C más frío en la carcasa), cal acumulada 0.67, ley de Sn de carga 0.62. El GN
de Reducción cae fuera de [P1, P99] en el 19 % de los escalones del lockbox. Las palancas que más se mueven en el lockbox son
justamente las del prescriptor (GN, exceso de O₂): el reentrenamiento progresivo es necesario para sostener los θ.

## 6. Límites

Curva de parsimonia con las palancas P3 (dosis/sobrante) y no con las definitivas (por eso `e7_07`); tercios de ~100 batches y
lockbox de 63 dan IC anchos; el modelo térmico de Fusión no se evaluó aquí.
