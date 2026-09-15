// ╔══════════════════════════════════════════════════════════════════════╗
// ║                        ══ COPYRIGHT NOTICE ══                        ║
// ║                    Code Master: Yair Camborda                        ║
// ║                   All Rights Reserved © Minsur SA                    ║
// ║                           Year 2024                                  ║
// ║----------------------------------------------------------------------║
// ║ Welcome to this flawless piece of art, also known as code. You are   ║
// ║ about to experience a program so refined, it's almost like it was    ║
// ║ coded by someone competent!                                          ║
// ║----------------------------------------------------------------------║
// ║ In the unforgiving wilderness of Rust, we handle memory leaks like   ║
// ║ a Jedi manages their emotions: We pretend they don't exist until     ║
// ║ they absolutely do. "May the Rust be with you"—or else.              ║
// ║----------------------------------------------------------------------║
// ║ Crafted with the precision of a Half-Life physics puzzle, this code  ║
// ║ runs tighter than a Naruto filler episode. Expect speeds so fast,    ║
// ║ Hollow Knight himself would want to dodge them.                      ║
// ║----------------------------------------------------------------------║
// ║ Here's the twist: despite its epic backstory of being resurrected    ║
// ║ from a previous disaster, this project is as stable as a D&D         ║
// ║ campaign run by a first-time dungeon master. Fear not, it’s          ║
// ║ definitely not going to crash and burn... probably.                  ║
// ║----------------------------------------------------------------------║
// ║ So, gear up, fellow coder. If this codebase were any more robust,    ║
// ║ it'd be a Dark Souls boss—unforgiving, mysterious, and rewarding     ║
// ║ only to the persistent.                                              ║
// ║----------------------------------------------------------------------║
// ║ Embrace the challenge, enjoy the bugs. May your coffee be strong     ║
// ║ and your compile times short.                                        ║
// ║                                                                      ║
// ║                      ~ Yair Camborda                                 ║
// ║                                                                      ║
// ║ PS: Remember, if something breaks, it's definitely by design.        ║
// ║ Just a feature waiting to be understood by mere mortals.             ║
// ╚══════════════════════════════════════════════════════════════════════╝

use crate::json_manager::JSONManager;
use crate::optimizer::start_optimization;
use log::{info, warn, LevelFilter};
use polars::frame::DataFrame;
use std::collections::HashMap;
use std::time::Instant;

#[cfg(feature = "use_f32")]
type Number = f32;
#[cfg(feature = "use_f64")]
type Number = f64;
type HmHmDf = (HashMap<String, Number>, HashMap<String, Number>, DataFrame);
mod fusion_energy;
pub mod fusion_matter;
pub mod helpers;
pub mod json_manager;
pub mod json_structs;
mod optimizer;
pub mod reduction_energy;
pub mod reduction_matter;

#[allow(dead_code)]
const JSON_DATA: &str = r#"{
  "tiempo_fusion": "6.00",
  "temperatura_inicial_escoria_red": "1200",
  "temperatura_metal_fus": 975.0
  "gas_natural_fusion": "1600",
  "enriquecimiento_o2_fusion": "1.0",
  "estequiometria_fusion": "108.9",
  "nivel_escoria_fusion": "600",
  "aire_post_combustion_fusion": "6000",
  "aire_infiltracion_fusion": "1000",
  "metal_drenado_fusion": "15",
  "carbon_perdido_tiro_fusion": "17.0",
  "temperatura_inicial_fusion_horno": "26",
  "temperatura_final_fusion_horno": "36.5",
  "temperatura_salida_gas_cooler_fusion": "205",
  "perc_escoria_fusion": "20",
  "factor_eficiencia_ca_o_fusion": "0.83",
  "tiempo_reduccion": "1.5",
  "gas_natural_reduccion": "1600",
  "enriquecimiento_o2_reduccion": "36.7088",
  "estequiometria_reduccion": "89.0",
  "nivel_escoria_reduccion": "600",
  "aire_post_combustion_reduccion": "6500",
  "aire_infiltracion_reduccion": "1000",
  "carbon_perdido_tiro_reduccion": "49.0",
  "temperatura_inicial_reduccion_horno": "36.5",
  "temperatura_final_reduccion_horno": "38.5",
  "temperatura_salida_gas_cooler_reduccion": "205",
  "perc_escoria_reduccion": "0.6",
  "factor_eficiencia_ca_o_reduccion": "0.89",
  "velocidad_opt": "1",
  "velocidad_opt_value": "27.6294",
  "densidad_escoria": 2.32,
  "diametro_interno": 3.692,
  "recirculantes": {
    "humos": [28.2608696, 8.0],
    "d_fe_fino": [0.0, 0.0],
    "d_fe_zarand": [12.0, 0.5],
    "oxidos": [0.0, 0.0],
    "d_cobre": [0.5, 0.0],
    "d_soda": [0.0, 0.0],
    "escoria_h_a": [0.0, 0.0],
    "escoria_h_r": [1.5, 0.5],
    "mineral_fe": [0.0, 0.0],
    "conc_secundario": [0.0, 0.0],
    "conc_cochas": [0.0, 0.0],
    "arena_silice": [0.0, 0.0],
    "otros": [15, 0.0]
  },
  "otras_configuraciones":{
    "cenizas_sio2": 40.0,
    "cenizas_al2o3": 30.0,
    "aire_atomizacion": 0.0,
    "humedad_aire": 0.0,
    "pureza_oxigeno": 98.0,
    "aportacion_mgo_escoria": 100.0,
    "aire_matriz": 1.0,
    "oxigeno_matriz": 1.0,
    "distribucion_gas_sno_fusion": 5.0,
    "distribucion_gas_sno_reduccion": 15.0,
    "distribucion_o2_libre_c2h6_fus": 0.0,
    "distribucion_o2_libre_o2_fus": 0.0,
    "distribucion_o2_libre_c2h6_red": 0.0,
    "distribucion_o2_libre_o2_red": 0.0,
    "lanza_ingreso_fusion": 0.0,
    "lanza_ingreso_reduccion": 0.0,
    "lanza_reacciona_fusion": 0.0,
    "lanza_reacciona_reduccion": 0.0,
    "codo_ingreso_fusion": 1.0,
    "codo_ingreso_reduccion": 1.0,
    "codo_reacciona_fusion": 1.0,
    "codo_reacciona_reduccion": 1.0,
    "distribucion_total_fe_metal": 6.0,
    "distribucion_total_fe_dross_fe": 10.0,
    "distribucion_total_fe_humos": 2.0,
    "distribucion_otros_metal_fus": 0.5,
    "distribucion_otros_metal_red": 1.0,
    "distribucion_otros_dross_fe_fus": 10.0,
    "distribucion_otros_dross_fe_red": 7.0,
    "distribucion_otros_escoria_fus": 75.0,
    "distribucion_otros_escoria_red": 77.0,
    "composicion_escoria_inicial_sn": 1.0,
    "composicion_escoria_inicial_fe": 13.21170,
    "composicion_escoria_inicial_sio2": 40.0,
    "composicion_escoria_inicial_al2o3": 13.0,
    "composicion_escoria_inicial_cao": 20.0,
    "composicion_escoria_inicial_mgo": 3.5,
    "distribucion_fe_fus_humos": 60.0,
    "distribucion_fe_fus_d_fe_metal": 65.0,
    "perc_sn_dross_fe_fus": 65.0,
    "perc_sn_dross_fe_red": 71.5,
    "factor_c_co_dross_fe": 1.0,
    "factor_c_co_normal": 0.0,
    "factor_c_co_fesn2": 30.0,
    "aire_lanza_gas_cooler_fus": 2000,
    "aire_lanza_gas_cooler_red": 2000,
    "flujo_agua": 358.137,
    "perdidas_fusion":248.587,
    "perdidas_reduccion": 468.872,
    "post_comb_fus": 5.0,
    "gas_cooler_fus": 3.0,
    "post_comb_red": 6.0,
    "gas_cooler_red": 4.0,
    "aire_lanza_fijo_fus": 1.0,
    "aire_lanza_fijo_red": 1.0,
    "aire_lanza_fus": 6950.0,
    "aire_lanza_red": 6950.0,
    "tipo_carbon_fus":2.0,
    "tipo_carbon_red":2.0,
  },
  "batch_id": "AL886",
  "cama_id": "AL179",
  "num_batches": "5"
}"#;

use crate::json_structs::TheoreticalInput;
use serde::{Deserialize, Serialize};
use std::env;
use std::net::Ipv4Addr;
use std::time::Duration;
use tokio::task;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use warp::{http::Response, Filter};

const VELOCITY_OPT_TIMEOUT_SECS: u64 = 180;
const MATTER_ERROR_TOLERANCE: f64 = 0.3;
const ENERGY_ERROR_TOLERANCE: f64 = 0.5;
const FALLBACK_REASON_TIMEOUT: &str = "velocity_opt_timeout_used_non_optimized_result";
const FALLBACK_REASON_NON_CONVERGENT: &str =
    "velocity_opt_non_convergent_used_non_optimized_result";
const FALLBACK_REASON_FAILED: &str = "velocity_opt_failed_used_non_optimized_result";

#[derive(Serialize, Deserialize)]
pub struct TheoreticalResponse {
    total_time: Number,
    // Fusion
    batch_id: String,
    cama_id: String,
    gas_natural_fusion: Number,
    enriquecimiento_o2_fusion: Number,
    estequiometria_fusion: Number,
    fusion_time: Number,
    nivel_escoria_fusion: Number,
    perc_escoria_fusion: Number,
    aire_post_combustion_fusion: Number,
    aire_infiltracion_fusion: Number,
    metal_drenado_fusion: Number,
    concentrado_fus: Number,
    carbon_fus: Number,
    aire_lanza_fus: Number,
    oxigeno_lanza_fus: Number,
    metal_fus: Number,
    escoria_fus: Number,
    dross_fe_fus: Number,
    humos_fus: Number,
    gases_chimenea_fus: Number,
    agua_gas_cooler_fus: Number,

    // Reduction
    gas_natural_reduccion: Number,
    enriquecimiento_o2_reduccion: Number,
    estequiometria_reduccion: Number,
    tiempo_reduccion: Number,
    nivel_escoria_reduccion: Number,
    perc_escoria_reduccion: Number,
    aire_post_combustion_reduccion: Number,
    aire_infiltracion_reduccion: Number,
    carbon_red: Number,
    aire_lanza_red: Number,
    oxigeno_lanza_red: Number,
    metal_red: Number,
    escoria_red: Number,
    dross_fe_red: Number,
    humos_red: Number,
    gases_chimenea_red: Number,
    agua_gas_cooler_red: Number,

    // Others
    velocidad_alimentacion: Number,
    velocidad_alimentacion_used: i8,
    basicidad: Number,
    rendimiento_fus: Number,
    rendimiento_red: Number,
    rendimiento_total: Number,
    sn_eh20: Number,
    carbon_peletizado_used: i8,
    carbon_peletizado: Number,
    carbon_fusion: Number,
    carbon_total_fus: Number,
    carbon_total_red: Number,
    input_balance_elements: HashMap<String, Number>,
    nivel_de_escoria: Number,
    porcentaje_feo_fusion: Number,
    porcentaje_feo_reduccion: Number,

    // New fields for errors
    no_convergence: bool,
    error_matter_fus: f64,
    error_matter_red: f64,
    error_energy_fus: f64,
    error_energy_red: f64,
    fallback_used: bool,
    fallback_reason: Option<String>,
    warning_message: Option<String>,
}

#[tokio::main]
async fn main() {
    env::set_var("POLARS_FMT_MAX_COLS", "-1");
    env::set_var("POLARS_FMT_MAX_ROWS", "-1");
    // Configuración del logger
    if cfg!(debug_assertions) {
        env_logger::builder()
            .filter_level(LevelFilter::Debug)
            .init();
    } else {
        env_logger::builder().filter_level(LevelFilter::Info).init();
    }

    let api = warp::post()
        .and(warp::path("api"))
        .and(warp::path("HttpTheoretical"))
        .and(warp::body::json())
        .and_then(handle_request);

    let port_key = "FUNCTIONS_CUSTOMHANDLER_PORT";
    let port: u16 = match env::var(port_key) {
        Ok(val) => val.parse().expect("Custom Handler port is not a number!"),
        Err(_) => 3000,
    };

    warp::serve(api).run((Ipv4Addr::LOCALHOST, port)).await
}

async fn run_optimization(input: TheoreticalInput) -> Result<TheoreticalResponse, String> {
    let mut json_manager = JSONManager::new(input)
        .await
        .map_err(|e| format!("Error creating JSON manager: {e}"))?;

    task::spawn_blocking(move || start_optimization(&mut json_manager))
        .await
        .map_err(|e| format!("Optimization task failed: {e}"))
}

fn is_result_acceptable(
    no_convergence: bool,
    error_matter_fus: f64,
    error_matter_red: f64,
    error_energy_fus: f64,
    error_energy_red: f64,
) -> bool {
    !no_convergence
        && error_matter_fus.abs() <= MATTER_ERROR_TOLERANCE
        && error_matter_red.abs() <= MATTER_ERROR_TOLERANCE
        && error_energy_fus.abs() <= ENERGY_ERROR_TOLERANCE
        && error_energy_red.abs() <= ENERGY_ERROR_TOLERANCE
}

fn is_response_acceptable(result: &TheoreticalResponse) -> bool {
    is_result_acceptable(
        result.no_convergence,
        result.error_matter_fus,
        result.error_matter_red,
        result.error_energy_fus,
        result.error_energy_red,
    )
}

fn fallback_warning_message(reason: &str) -> String {
    match reason {
        FALLBACK_REASON_TIMEOUT => format!(
            "No se logro converger con velocidad de optimizacion dentro de {} segundos. Se devolvio el resultado calculado sin velocidad de optimizacion. Revise duracion de fusion, velocidad de alimentacion y parametros asociados.",
            VELOCITY_OPT_TIMEOUT_SECS
        ),
        FALLBACK_REASON_NON_CONVERGENT => "La corrida con velocidad de optimizacion termino sin convergencia o con errores fuera de tolerancia. Se devolvio el resultado calculado sin velocidad de optimizacion. Revise duracion de fusion, velocidad de alimentacion y parametros asociados.".to_string(),
        FALLBACK_REASON_FAILED => "La corrida con velocidad de optimizacion fallo antes de converger. Se devolvio el resultado calculado sin velocidad de optimizacion. Revise duracion de fusion, velocidad de alimentacion y parametros asociados.".to_string(),
        _ => "Se devolvio el resultado calculado sin velocidad de optimizacion por una condicion de fallback no clasificada.".to_string(),
    }
}

fn apply_fallback_metadata(
    mut fallback_result: TheoreticalResponse,
    reason: &str,
) -> TheoreticalResponse {
    fallback_result.fallback_used = true;
    fallback_result.fallback_reason = Some(reason.to_string());
    fallback_result.warning_message = Some(fallback_warning_message(reason));
    fallback_result
}

async fn await_fallback(
    fallback_handle: JoinHandle<Result<TheoreticalResponse, String>>,
) -> Result<TheoreticalResponse, String> {
    fallback_handle
        .await
        .map_err(|e| format!("Fallback task failed: {e}"))?
}

async fn run_with_velocity_fallback(
    input: TheoreticalInput,
) -> Result<TheoreticalResponse, String> {
    if input.velocidad_opt != 1 {
        return run_optimization(input).await;
    }

    let optimized_input = input.clone();
    let mut fallback_input = input;
    fallback_input.velocidad_opt = 0;

    let fallback_handle = tokio::spawn(run_optimization(fallback_input));
    let optimized_handle = tokio::spawn(run_optimization(optimized_input));

    match timeout(
        Duration::from_secs(VELOCITY_OPT_TIMEOUT_SECS),
        optimized_handle,
    )
    .await
    {
        Ok(optimized_join) => {
            match optimized_join.map_err(|e| format!("Optimized task failed: {e}"))? {
                Ok(optimized_result) if is_response_acceptable(&optimized_result) => {
                    fallback_handle.abort();
                    Ok(optimized_result)
                }
                Ok(_) => {
                    let fallback_result = await_fallback(fallback_handle).await?;
                    Ok(apply_fallback_metadata(
                        fallback_result,
                        FALLBACK_REASON_NON_CONVERGENT,
                    ))
                }
                Err(_) => {
                    let fallback_result = await_fallback(fallback_handle).await?;
                    Ok(apply_fallback_metadata(
                        fallback_result,
                        FALLBACK_REASON_FAILED,
                    ))
                }
            }
        }
        Err(_) => {
            let fallback_result = await_fallback(fallback_handle).await?;
            Ok(apply_fallback_metadata(
                fallback_result,
                FALLBACK_REASON_TIMEOUT,
            ))
        }
    }
}

async fn handle_request(input: TheoreticalInput) -> Result<impl warp::Reply, warp::Rejection> {
    let start_time = Instant::now();
    let mut result = match run_with_velocity_fallback(input).await {
        Ok(result) => result,
        Err(e) => return Ok(Response::builder().status(500).body(e)),
    };
    let total_duration = start_time.elapsed().as_secs_f64();

    // Loggear los tiempos
    info!("Total time: {:.2} seconds", total_duration);
    if result.fallback_used {
        match result.fallback_reason.as_deref() {
            Some(FALLBACK_REASON_TIMEOUT) => warn!(
                "Velocity optimization timed out after {} seconds. Returning fallback result without velocity optimization.",
                VELOCITY_OPT_TIMEOUT_SECS
            ),
            Some(FALLBACK_REASON_NON_CONVERGENT) => warn!(
                "Velocity optimization returned a non-convergent result. Returning fallback result without velocity optimization."
            ),
            Some(FALLBACK_REASON_FAILED) => warn!(
                "Velocity optimization failed before converging. Returning fallback result without velocity optimization."
            ),
            Some(reason) => warn!(
                "Velocity optimization fallback used. Returning fallback result without velocity optimization. reason={}",
                reason
            ),
            None => warn!(
                "Velocity optimization fallback used. Returning fallback result without velocity optimization."
            ),
        }
    }
    if result.no_convergence {
        warn!(
            "Final result did not converge. Errors: error_matter_fus={:.3}, error_matter_red={:.3}, error_energy_fus={:.3}, error_energy_red={:.3}",
            result.error_matter_fus,
            result.error_matter_red,
            result.error_energy_fus,
            result.error_energy_red
        );
    }

    result.total_time = total_duration;

    let result_json = match serde_json::to_string(&result) {
        Ok(json) => json,
        Err(e) => return Ok(Response::builder().status(500).body(e.to_string())),
    };

    Ok(Response::builder()
        .header("Content-Type", "application/json")
        .body(result_json))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_converged_result_within_tolerance() {
        assert!(is_result_acceptable(false, -0.2, 0.1, 0.0, -0.4));
    }

    #[test]
    fn rejects_result_marked_as_non_convergent() {
        assert!(!is_result_acceptable(true, -0.2, 0.1, 0.0, -0.4));
    }

    #[test]
    fn rejects_result_outside_error_tolerance() {
        assert!(!is_result_acceptable(false, 0.31, 0.1, 0.0, -0.4));
        assert!(!is_result_acceptable(false, -0.2, 0.1, 0.51, -0.4));
    }

    #[test]
    fn builds_specific_warning_messages() {
        assert!(
            fallback_warning_message(FALLBACK_REASON_TIMEOUT).contains("dentro de 180 segundos")
        );
        assert!(fallback_warning_message(FALLBACK_REASON_NON_CONVERGENT)
            .contains("termino sin convergencia"));
        assert!(fallback_warning_message(FALLBACK_REASON_FAILED).contains("fallo"));
    }
}
