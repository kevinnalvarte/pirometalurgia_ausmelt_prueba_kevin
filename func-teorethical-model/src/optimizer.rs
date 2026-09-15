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
use crate::fusion_energy::FusionEnergy;
use crate::fusion_matter::FusionMatter;
use crate::helpers::DataFrameCustomExtensions;
use crate::json_manager::JSONManager;
use crate::reduction_energy::ReductionEnergy;
use crate::reduction_matter::ReductionMatter;
use crate::{Number, TheoreticalResponse};
use log::debug;

fn calc_temperature(json_manager: &mut JSONManager) {
    let max_iterations = 100;
    let mut iteration = 0;

    while iteration < max_iterations {
        // Recalcular objetos una vez por iteración
        json_manager.no_convergence = false;
        let fusion_matter = FusionMatter::new(json_manager).unwrap();
        let fusion_energy = FusionEnergy::new(json_manager, &fusion_matter).unwrap();
        let reduction_matter = ReductionMatter::new(json_manager, &fusion_matter).unwrap();
        let reduction_energy = ReductionEnergy::new(
            json_manager,
            &fusion_matter,
            &reduction_matter,
            &fusion_energy,
        )
        .unwrap();

        // Recalcular errores
        let (error_matter_fus, error_matter_red, error_energy_fus, error_energy_red) = get_errors(
            json_manager,
            &fusion_matter,
            &fusion_energy,
            &reduction_matter,
            &reduction_energy,
        );

        json_manager.error_matter_fus = error_matter_fus;
        json_manager.error_matter_red = error_matter_red;
        json_manager.error_energy_fus = error_energy_fus;
        json_manager.error_energy_red = error_energy_red;

        if error_matter_fus.abs() <= 0.3
            && error_matter_red.abs() <= 0.3
            && error_energy_fus.abs() <= 0.5
            && error_energy_red.abs() <= 0.5
        {
            break;
        }

        iteration += 1;

        // Ajuste de parámetros basado en las diferencias

        if json_manager.theoretical_input.otras_configuraciones["aire_lanza_fijo_fus"] == 1.0 {
            let air_lance_fus =
                json_manager.theoretical_input.otras_configuraciones["aire_lanza_fus"];
            let diff = air_lance_fus - json_manager.obj_air_result_nm3_h_fus;
            json_manager.obj_o2_fus_enrichment -= diff / 500.0;
        }

        if json_manager.theoretical_input.otras_configuraciones["aire_lanza_fijo_red"] == 1.0 {
            let air_lance_red =
                json_manager.theoretical_input.otras_configuraciones["aire_lanza_red"];
            let diff = air_lance_red - json_manager.obj_air_result_nm3_h_red;
            json_manager.obj_o2_red_enrichment -= diff / 500.0;
        }

        // Optimización de carbono en fusión
        let diff_fusion_carbon = json_manager.free_c_fus - fusion_matter.reactions[18]["c"];
        json_manager.obj_carbon_fus -= diff_fusion_carbon;

        // Optimización de escoria en fusión
        let diff_fusion_dump = fusion_matter.hm_second_dump["sn"] - json_manager.sn_dump_end_fusion;
        json_manager.obj_distribution_sno_dump_fus -= diff_fusion_dump / 5.0;

        // Optimización de FeO en fusión
        let diff_fusion_feo = fusion_matter
            .df_metal
            .get_conditional_value("elements", "fe", "tm")
            .unwrap()
            + fusion_matter
                .df_dross_fe
                .get_conditional_value("elements", "fe", "tm")
                .unwrap()
            - fusion_matter.reactions[10]["fe"];
        json_manager.obj_dist_feo_fe_fus += diff_fusion_feo * 10.0;

        // Optimización de carbono en reducción
        let diff_reduction_carbon = json_manager.free_c_red - reduction_matter.reactions[16]["c"];
        json_manager.obj_carbon_red -= diff_reduction_carbon;

        // Optimización de escoria en reducción
        let diff_reduction_dump =
            reduction_matter.hm_second_dump["sn"] - json_manager.sn_dump_end_reduction;
        json_manager.obj_distribution_sno_dump_red -= diff_reduction_dump / 5.0;

        // Optimización de FeO en reducción
        let diff_reduction_feo = reduction_matter
            .df_metal
            .get_conditional_value("elements", "fe", "tm")
            .unwrap()
            + reduction_matter
                .df_dross_fe
                .get_conditional_value("elements", "fe", "tm")
                .unwrap()
            - reduction_matter.reactions[8]["fe"];
        json_manager.obj_dist_feo_fe_red += diff_reduction_feo * 10.0;

        // Optimización de energía en fusión
        let diff_fusion_stove_energy =
            fusion_energy.total_heat_input - fusion_energy.total_heat_output;
        json_manager.obj_stove_energy_fus += diff_fusion_stove_energy / 100.0;

        let diff_fusion_post_comb_energy = fusion_energy.total_input_heat_post_combustion
            - fusion_energy.total_output_heat_post_combustion;
        json_manager.obj_post_comb_energy_fus += diff_fusion_post_comb_energy / 100.0;

        let diff_fusion_gas_cooler_energy =
            fusion_energy.total_input_heat_gas_cooler - fusion_energy.total_output_heat_gas_cooler;
        json_manager.obj_water_consumption_fus += diff_fusion_gas_cooler_energy / 10.0;

        // Optimización de energía en reducción
        let diff_reduction_stove_energy =
            reduction_energy.total_heat_input - reduction_energy.total_heat_output;
        json_manager.obj_stove_energy_red += diff_reduction_stove_energy / 50.0;

        let diff_reduction_post_comb_energy = reduction_energy.total_input_heat_post_combustion
            - reduction_energy.total_output_heat_post_combustion;
        json_manager.obj_post_comb_energy_red += diff_reduction_post_comb_energy / 50.0;

        let diff_reduction_gas_cooler_energy = reduction_energy.total_input_heat_gas_cooler
            - reduction_energy.total_output_heat_gas_cooler;
        json_manager.obj_water_consumption_red += diff_reduction_gas_cooler_energy / 10.0;
    }

    if iteration == max_iterations {
        json_manager.no_convergence = true;
    }
}

fn get_errors(
    json_manager: &mut JSONManager,
    fusion_matter: &FusionMatter,
    fusion_energy: &FusionEnergy,
    reduction_matter: &ReductionMatter,
    reduction_energy: &ReductionEnergy,
) -> (Number, Number, Number, Number) {
    let error_matter_fus: Number;
    {
        // input
        let concentrate = json_manager.hashmap_cama["tmh"];
        let ox_calc = json_manager.theoretical_input.recirculantes["otros"][0];
        let recirculated = json_manager.recirculated_tmh
            + json_manager.theoretical_input.otras_configuraciones["aportacion_mgo_escoria"]
                / 1000.
            - ox_calc;
        let carbon = json_manager.obj_carbon_fus;

        let gas = fusion_matter
            .df_combustion_income
            .get_conditional_value("components", "total", "combustible")
            .unwrap();
        let air = fusion_matter
            .df_combustion_income
            .get_conditional_value("components", "total", "a_atomized")
            .unwrap()
            + fusion_matter
                .df_combustion_income
                .get_conditional_value("components", "total", "air")
                .unwrap();

        let oxygen = fusion_matter
            .df_combustion_income
            .get_conditional_value("components", "total", "oxygen")
            .unwrap();

        let dump = json_manager.initial_weight_dump;
        let air_post =
            (json_manager.air_n2_fus + json_manager.air_o2_fus + json_manager.air_h2o_fus)
                * json_manager.air_post_combustion_fus_m3
                / json_manager.total_air_fus_m3;
        let air_inf =
            (json_manager.air_n2_fus + json_manager.air_o2_fus + json_manager.air_h2o_fus)
                * json_manager.air_infiltration_fus_m3
                / json_manager.total_air_fus_m3;

        let input = concentrate
            + ox_calc
            + recirculated
            + carbon
            + gas
            + air
            + oxygen
            + dump
            + air_post
            + air_inf;

        // output

        let drained_total = fusion_matter
            .df_metal
            .get_conditional_value("elements", "total", "tm")
            .unwrap();
        let dump_total = fusion_matter
            .df_dump
            .get_conditional_value("elements", "total", "tm")
            .unwrap();
        let dross_total = fusion_matter
            .df_dross_fe
            .get_conditional_value("elements", "total", "tm")
            .unwrap();
        let fumes_total = fusion_matter
            .hm_post_combustion_fumes
            .values()
            .sum::<Number>();
        let gases_total = fusion_matter.hm_gases_post["total"];

        let output = drained_total + dump_total + dross_total + fumes_total + gases_total;

        // result
        error_matter_fus = input - output;
    }

    let error_matter_red: Number;
    {
        // input
        let stove_metal = reduction_matter
            .df_income_charge
            .get_conditional_value("molecules", "tmh", "metal")
            .unwrap();
        let dump = reduction_matter
            .df_income_charge
            .get_conditional_value("molecules", "tmh", "dump")
            .unwrap();
        let carbon = json_manager.obj_carbon_red;
        let gas = reduction_matter
            .df_combustion_income
            .get_conditional_value("components", "total", "combustible")
            .unwrap();
        let air = reduction_matter
            .df_combustion_income
            .get_conditional_value("components", "total", "a_atomized")
            .unwrap()
            + reduction_matter
                .df_combustion_income
                .get_conditional_value("components", "total", "air")
                .unwrap();

        let oxygen = reduction_matter
            .df_combustion_income
            .get_conditional_value("components", "total", "oxygen")
            .unwrap();

        let air_post =
            (json_manager.air_n2_red + json_manager.air_o2_red + json_manager.air_h2o_red)
                * json_manager.air_post_combustion_red_m3
                / json_manager.total_air_red_m3;
        let air_inf =
            (json_manager.air_n2_red + json_manager.air_o2_red + json_manager.air_h2o_red)
                * json_manager.air_infiltration_red_m3
                / json_manager.total_air_red_m3;

        let input = stove_metal + dump + carbon + gas + air + oxygen + air_post + air_inf;

        // output

        let drained_total = reduction_matter
            .df_metal
            .get_conditional_value("elements", "total", "tm")
            .unwrap();
        let dump_total = reduction_matter
            .df_dump
            .get_conditional_value("elements", "total", "tm")
            .unwrap();
        let dross_total = reduction_matter
            .df_dross_fe
            .get_conditional_value("elements", "total", "tm")
            .unwrap();
        let fumes_total = reduction_matter
            .hm_post_combustion_fumes
            .values()
            .sum::<Number>();
        let gases_total = reduction_matter.hm_gases_post["total"];

        let output = drained_total + dump_total + dross_total + fumes_total + gases_total;

        error_matter_red = input - output;
    }

    let error_energy_fus: Number;
    {
        let hm_heat_input = &fusion_energy.hm_heat_input;
        let hm_input_heat_post_combustion = &fusion_energy.hm_input_heat_post_combustion;

        // input
        let gas = hm_heat_input["mcal_petroleum_fuel"];
        let petroleum = hm_heat_input["mcal_sensible_petroleum"];
        let air_and_oxy = hm_heat_input["mcal_sense_air"]
            + hm_heat_input["mcal_sense_oxygen"]
            + hm_input_heat_post_combustion["mcal_air_heat_sense"];
        let dump = hm_heat_input["mcal_sense_init_dump"];
        let exo_reactions = hm_heat_input["mcal_reactions"];
        let exo_reactions_post = hm_input_heat_post_combustion["mcal_reactions_heat_sense"];
        let input = gas + petroleum + air_and_oxy + dump + exo_reactions + exo_reactions_post;

        // output
        let hm_total_heat_output = &fusion_energy.hm_total_heat_output;
        let hm_output_heat_post_combustion = &fusion_energy.hm_output_heat_post_combustion;

        let metal_heat_sense = hm_total_heat_output["mcal_metal_heat_sense"];
        let metal_dump_sense = hm_total_heat_output["mcal_dump_heat_sense"];
        let mcal_dross_heat_sense = hm_total_heat_output["mcal_dross_heat_sense"];
        let mcal_fumes_heat_sense = hm_output_heat_post_combustion["mcal_fumes_heat_sense"];
        let mcal_gases_heat_sense = hm_output_heat_post_combustion["mcal_gases_heat_sense"];
        let mcal_reactions = hm_total_heat_output["mcal_reactions"];
        let mcal_latent_heat = hm_total_heat_output["mcal_latent_heat"];
        let mcal_refrigeration = hm_total_heat_output["mcal_refrigeration"];
        let lost =
            hm_total_heat_output["mcal_heat_lost"] + hm_output_heat_post_combustion["mcal_loses"];

        let output = metal_heat_sense
            + metal_dump_sense
            + mcal_dross_heat_sense
            + mcal_fumes_heat_sense
            + mcal_gases_heat_sense
            + mcal_reactions
            + mcal_latent_heat
            + mcal_refrigeration
            + lost;

        error_energy_fus = input - output;
    }

    let error_energy_red: Number;
    {
        let hm_heat_input = &reduction_energy.hm_heat_input;
        let hm_input_heat_post_combustion = &reduction_energy.hm_input_heat_post_combustion;

        // input
        let gas = hm_heat_input["mcal_petroleum_combustion"];
        let petroleum = hm_heat_input["mcal_heat_sense_petroleum"];
        let air_and_oxy = hm_heat_input["mcal_sense_air"]
            + hm_heat_input["mcal_sense_oxygen"]
            + hm_input_heat_post_combustion["mcal_air_heat_sense"];
        let metal = hm_heat_input["mcal_metal_heat_sense"];
        let dump = hm_heat_input["mcal_dump_heat_sense"];
        let exo_reactions = hm_heat_input["mcal_reactions"];
        let exo_reactions_post = hm_input_heat_post_combustion["mcal_reactions_heat_sense"];
        let input =
            gas + petroleum + air_and_oxy + metal + dump + exo_reactions + exo_reactions_post;

        // output
        let hm_total_heat_output = &reduction_energy.hm_total_heat_output;
        let hm_output_heat_post_combustion = &reduction_energy.hm_output_heat_post_combustion;

        let metal_heat_sense = hm_total_heat_output["mcal_metal_heat_sense"];
        let metal_dump_sense = hm_total_heat_output["mcal_dump_heat_sense"];
        let mcal_dross_heat_sense = hm_total_heat_output["mcal_dross_heat_sense"];
        let mcal_fumes_heat_sense = hm_output_heat_post_combustion["mcal_fumes_heat_sense"];
        let mcal_gases_heat_sense = hm_output_heat_post_combustion["mcal_gases_heat_sense"];
        let mcal_reactions = hm_total_heat_output["mcal_reactions"];
        let mcal_latent_heat = hm_total_heat_output["mcal_latent_heat"];
        let mcal_refrigeration = hm_total_heat_output["mcal_refrigeration"];
        let lost =
            hm_total_heat_output["mcal_heat_lost"] + hm_output_heat_post_combustion["mcal_loses"];

        let output = metal_heat_sense
            + metal_dump_sense
            + mcal_dross_heat_sense
            + mcal_fumes_heat_sense
            + mcal_gases_heat_sense
            + mcal_reactions
            + mcal_latent_heat
            + mcal_refrigeration
            + lost;

        error_energy_red = input - output;
    }

    (
        error_matter_fus,
        error_matter_red,
        error_energy_fus,
        error_energy_red,
    )
}

fn calc_combustible(json_manager: &mut JSONManager) {
    let z = if json_manager.theoretical_input.otras_configuraciones["tipo_carbon_fus"] == 1.0 {
        3.0
    } else {
        1.0
    };
    if json_manager.obj_air_result_nm3_h_fus < 0.0 || json_manager.obj_air_result_nm3_h_red < 0.0 {
        calc_temperature(json_manager);
    }

    let max_iterations = 100;
    let mut iteration = 0;

    while iteration < max_iterations {
        let temp_metal_fus = json_manager.theoretical_input.temperatura_metal_fus;
        let error = temp_metal_fus - json_manager.obj_stove_energy_fus;

        if error.abs() <= 0.4 {
            break;
        }

        let factor_corr = if error.abs() > 40. {
            1.3
        } else if error.abs() > 5. {
            1.15
        } else {
            1.0
        };
        json_manager.obj_gas_fus += error * factor_corr / z;
        calc_temperature(json_manager);
        iteration += 1;
    }

    iteration = 0;

    while iteration < max_iterations {
        let dum_temp_red = json_manager
            .theoretical_input
            .temperatura_inicial_escoria_red;
        let error = dum_temp_red - json_manager.obj_stove_energy_red;

        if error.abs() <= 0.4 {
            break;
        }

        let factor_corr = if error.abs() > 50. {
            1.6
        } else if error.abs() > 5. {
            1.5
        } else {
            1.0
        };
        json_manager.obj_gas_red += error * factor_corr / z;
        calc_temperature(json_manager);
        iteration += 1;
    }
}

#[allow(dead_code)]
fn debug_process(
    jsonmanager: &mut JSONManager,
    fusion_matter: &FusionMatter,
    reduction_matter: &ReductionMatter,
) {
    let fusion_energy = FusionEnergy::new(jsonmanager, fusion_matter).unwrap();
    let reduction_energy =
        ReductionEnergy::new(jsonmanager, fusion_matter, reduction_matter, &fusion_energy).unwrap();
    debug!("FusionMatter: {:?}", fusion_matter);
    debug!("{}", "*".repeat(50));
    debug!("ReductionMatter: {:?}", reduction_matter);
    debug!("{}", "*".repeat(50));
    debug!("ReductionEnergy: {:?}", reduction_energy);
    debug!("{}", "*".repeat(50));
    debug!("JSONManager: {:?}", jsonmanager);
    debug!("{}", "*".repeat(50));
    debug!("FusionEnergy: {:?}", fusion_energy);
}

pub fn start_optimization(json_manager: &mut JSONManager) -> TheoreticalResponse {
    calc_temperature(json_manager);
    calc_combustible(json_manager);

    if json_manager.theoretical_input.velocidad_opt == 1 {
        let concentrate = json_manager.hashmap_cama["tmh"];
        let limestone = &json_manager
            .df_limestone_weight
            .get_conditional_value("caliza", "total", "tmh")
            .unwrap();
        let recirculated = json_manager.recirculated_tmh;
        let carbon = json_manager.obj_carbon_fus
            * 1000.
            * (100. / (100. - json_manager.theoretical_input.carbon_perdido_tiro_fusion));
        let charge = concentrate + limestone + recirculated + carbon / 1000.;
        let new_fusion_time = charge / json_manager.theoretical_input.velocidad_opt_value;
        json_manager.fusion_time = new_fusion_time;
        calc_temperature(json_manager);
        calc_combustible(json_manager);
    }

    let fusion_matter = FusionMatter::new(json_manager).unwrap();
    let reduction_matter = ReductionMatter::new(json_manager, &fusion_matter).unwrap();
    if log::log_enabled!(log::Level::Debug) {
        debug_process(json_manager, &fusion_matter, &reduction_matter);
    }

    let concentrate_fus = json_manager.hashmap_cama["tmh"] / json_manager.fusion_time;
    let carbon_fus = json_manager.obj_carbon_fus
        * 1000.
        * (100. / (100. - json_manager.theoretical_input.carbon_perdido_tiro_fusion))
        / json_manager.fusion_time;
    let metal_fus = fusion_matter
        .df_metal
        .get_conditional_value("elements", "total", "tm")
        .unwrap();
    let dump_fus = fusion_matter
        .df_dump
        .get_conditional_value("elements", "total", "tm")
        .unwrap();
    let dross_fe_fus = fusion_matter
        .df_dross_fe
        .get_conditional_value("elements", "total", "tm")
        .unwrap();
    let fumes_fus: Number = fusion_matter.hm_post_combustion_fumes.values().sum();
    let gases_chimney_fus = fusion_matter
        .df_gases_chimney
        .get_conditional_value("elements", "total", "nm3_h")
        .unwrap();
    let carbon_red = json_manager.obj_carbon_red
        * 1000.
        * (100. / (100. - json_manager.theoretical_input.carbon_perdido_tiro_reduccion))
        / json_manager.reduction_time;
    let metal_red = reduction_matter
        .df_metal
        .get_conditional_value("elements", "total", "tm")
        .unwrap();
    let dump_red_drained = reduction_matter
        .df_dump
        .get_conditional_value("elements", "total", "tm")
        .unwrap()
        - json_manager.initial_weight_dump;

    let dump_red_total = reduction_matter
        .df_dump
        .get_conditional_value("elements", "total", "tm")
        .unwrap();
    let weight_at_reduction_level =
        json_manager.theoretical_input.diametro_interno.powi(2) * std::f64::consts::PI / 4.0
            * json_manager.theoretical_input.nivel_escoria_reduccion
            / 1000.0
            * json_manager.theoretical_input.densidad_escoria;
    let dump_level = json_manager.theoretical_input.nivel_escoria_reduccion
        / weight_at_reduction_level
        * dump_red_total
        / 1000.0;

    let dump_fus_perc = fusion_matter
        .df_dump
        .get_conditional_value("elements", "feo", "perc")
        .unwrap()
        * 55.85
        / (55.85 + 16.0)
        * (71.8 / 55.8);
    let dump_red_perc = reduction_matter
        .df_dump
        .get_conditional_value("elements", "feo", "perc")
        .unwrap()
        * 55.85
        / (55.85 + 16.0)
        * (71.8 / 55.8);

    let dross_fe_red = reduction_matter
        .df_dross_fe
        .get_conditional_value("elements", "total", "tm")
        .unwrap();
    let fumes_red: Number = reduction_matter.hm_post_combustion_fumes.values().sum();
    let gases_chimney_red = reduction_matter
        .df_gases_chimney
        .get_conditional_value("elements", "total", "nm3_h")
        .unwrap();

    let basicity = reduction_matter.hm_second_dump["cao"] / reduction_matter.hm_second_dump["sio2"]
        * json_manager
            .theoretical_input
            .factor_eficiencia_ca_o_reduccion;
    let hashmap_charge = fusion_matter
        .df_income_charge
        .generate_hashmap_str_number("molecules", "total")
        .unwrap();
    let sn_total_input = hashmap_charge["sno2"] * 118.7 / (118.7 + 32.)
        + hashmap_charge["sno"] * 118.7 / (118.7 + 16.)
        + hashmap_charge["sns"] * 118.7 / (118.7 + 32.)
        + hashmap_charge["sn"]
        + hashmap_charge["fesn2"] * (118.7 * 2.) / (118.7 * 2. + 55.85);
    let sn_eh20 = fusion_matter.hm_second_dump["sn"];

    let carbon_fusion = carbon_fus * json_manager.fusion_time
        - if json_manager.theoretical_input.carbon_peletizado == 1 {
            json_manager.theoretical_input.carbon_peletizado_value
        } else {
            0.0
        };

    let carbon_total_fus = carbon_fus * json_manager.fusion_time;

    TheoreticalResponse {
        total_time: 0.0,
        // Fusion
        batch_id: json_manager.theoretical_input.batch_id.clone(),
        cama_id: json_manager.theoretical_input.cama_id.clone(),
        gas_natural_fusion: json_manager.obj_gas_fus,
        enriquecimiento_o2_fusion: json_manager.obj_o2_fus_enrichment,
        estequiometria_fusion: json_manager.obj_fus_stoichiometry,
        fusion_time: json_manager.fusion_time,
        nivel_escoria_fusion: json_manager.theoretical_input.nivel_escoria_fusion,
        perc_escoria_fusion: json_manager.theoretical_input.perc_escoria_fusion,
        aire_post_combustion_fusion: json_manager.theoretical_input.aire_post_combustion_fusion,
        aire_infiltracion_fusion: json_manager.theoretical_input.aire_infiltracion_fusion,
        metal_drenado_fusion: json_manager.metal_drained,
        concentrado_fus: concentrate_fus,
        carbon_fus,
        aire_lanza_fus: json_manager.obj_air_result_nm3_h_fus,
        oxigeno_lanza_fus: json_manager.obj_oxygen_result_nm3_h_fus,
        metal_fus,
        escoria_fus: dump_fus,
        dross_fe_fus,
        humos_fus: fumes_fus,
        gases_chimenea_fus: gases_chimney_fus,
        agua_gas_cooler_fus: json_manager.obj_water_consumption_fus,
        carbon_fusion: carbon_fusion,
        // Reduction
        gas_natural_reduccion: json_manager.obj_gas_red,
        enriquecimiento_o2_reduccion: json_manager.obj_o2_red_enrichment,
        estequiometria_reduccion: json_manager.obj_red_stoichiometry,
        tiempo_reduccion: json_manager.reduction_time,
        nivel_escoria_reduccion: json_manager.theoretical_input.nivel_escoria_reduccion,
        perc_escoria_reduccion: json_manager.theoretical_input.perc_escoria_reduccion,
        aire_post_combustion_reduccion: json_manager
            .theoretical_input
            .aire_post_combustion_reduccion,
        aire_infiltracion_reduccion: json_manager.theoretical_input.aire_infiltracion_reduccion,
        carbon_red,
        aire_lanza_red: json_manager.obj_air_result_nm3_h_red,
        oxigeno_lanza_red: json_manager.obj_oxygen_result_nm3_h_red,
        metal_red,
        escoria_red: dump_red_drained,
        dross_fe_red,
        humos_red: fumes_red,
        gases_chimenea_red: gases_chimney_red,
        agua_gas_cooler_red: json_manager.obj_water_consumption_red,

        // Others
        velocidad_alimentacion_used: json_manager.theoretical_input.velocidad_opt,
        velocidad_alimentacion: json_manager.theoretical_input.velocidad_opt_value,
        carbon_peletizado_used: json_manager.theoretical_input.carbon_peletizado,
        carbon_peletizado: json_manager.theoretical_input.carbon_peletizado_value,
        basicidad: basicity,
        rendimiento_fus: metal_fus / sn_total_input,
        rendimiento_red: metal_red / sn_total_input,
        rendimiento_total: (metal_fus + metal_red) / sn_total_input,
        sn_eh20,
        carbon_total_fus,
        carbon_total_red: carbon_red * json_manager.reduction_time,
        input_balance_elements: fusion_matter.hm_balance_elements,
        nivel_de_escoria: dump_level,
        porcentaje_feo_fusion: dump_fus_perc,
        porcentaje_feo_reduccion: dump_red_perc,

        // Errors
        no_convergence: json_manager.no_convergence,
        error_matter_fus: json_manager.error_matter_fus,
        error_matter_red: json_manager.error_matter_red,
        error_energy_fus: json_manager.error_energy_fus,
        error_energy_red: json_manager.error_energy_red,
        fallback_used: false,
        fallback_reason: None,
        warning_message: None,
    }
}
