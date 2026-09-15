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

use crate::helpers::{DataFrameCustomExtensions, OptimizationHelpers};
use crate::json_structs::{mw, TheoreticalInput};
use crate::Number;
use log::debug;
use polars::export::num::Pow;
use polars::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::io::Cursor;

pub struct JSONManager {
    pub theoretical_input: TheoreticalInput,
    pub hashmap_cama: HashMap<String, Number>,
    pub df_cama: DataFrame,
    pub df_mineralogy_limestone: DataFrame,
    pub df_chemistry_limestone: DataFrame,
    pub df_limestone_weight: DataFrame,
    pub df_recirculated_mineralogy: DataFrame,
    pub df_recirculated_chemistry: DataFrame,
    pub df_carbon_feed: DataFrame,
    pub df_fuel_parameters: DataFrame,
    pub df_heating: DataFrame,
    pub df_fusion_minor_elements: DataFrame,
    pub df_reduction_minor_elements: DataFrame,
    pub df_energy_elements: DataFrame,
    pub df_energy_equations: DataFrame,
    pub df_energy_water: DataFrame,

    pub recirculated_tmh: Number,
    pub recirculated_humidity: Number,
    pub initial_weight_dump: Number,
    pub free_oxygen_fus: Number,
    pub free_oxygen_red: Number,
    pub fuel_factor: Number,
    pub fuel_heating: Number,
    pub time_heating: Number,
    pub heating_stoichiometry: Number,
    pub c_air_fus: Number,
    pub metal_drained: Number,

    pub sn_dump_end_fusion: Number,
    pub sn_dump_end_reduction: Number,
    pub fusion_time: Number,
    pub reduction_time: Number,

    pub air_post_combustion_fus: Number,
    pub air_infiltration_fus: Number,
    pub total_air_fus: Number,
    pub total_air_red: Number,
    pub air_post_combustion_fus_m3: Number,
    pub air_infiltration_fus_m3: Number,
    pub air_post_combustion_red_m3: Number,
    pub air_infiltration_red_m3: Number,
    pub total_air_fus_m3: Number,
    pub total_air_red_m3: Number,
    pub co_o5_fus: Number,
    pub co_o5_red: Number,

    pub air_post_combustion_red: Number,
    pub air_infiltration_red: Number,

    pub air_humidity: Number,
    pub air_o2_fus: Number,
    pub air_n2_fus: Number,
    pub air_h2o_fus: Number,
    pub air_o2_red: Number,
    pub air_n2_red: Number,
    pub air_h2o_red: Number,
    pub air_o2_others_fus: Number,
    pub air_n2_others_fus: Number,
    pub air_h2o_others_fus: Number,
    pub air_o2_others_red: Number,
    pub air_n2_others_red: Number,
    pub air_h2o_others_red: Number,
    pub total_dist_tm_fe: Number,
    pub free_c_fus: Number,
    pub free_c_red: Number,

    pub obj_carbon_red: Number,
    pub obj_carbon_fus: Number,
    pub obj_gas_fus: Number,
    pub obj_gas_red: Number,
    pub obj_fus_stoichiometry: Number,
    pub obj_red_stoichiometry: Number,
    pub obj_o2_fus_enrichment: Number,
    pub obj_o2_red_enrichment: Number,
    pub obj_air_result_nm3_h_fus: Number,
    pub obj_oxygen_result_nm3_h_fus: Number,
    pub obj_air_result_nm3_h_red: Number,
    pub obj_oxygen_result_nm3_h_red: Number,
    pub obj_distribution_sno_dump_fus: Number,
    pub obj_distribution_sno_dump_red: Number,
    pub obj_dist_feo_fe_fus: Number,
    pub obj_dist_feo_fe_red: Number,
    pub obj_dist_c_co_fus: Number,
    pub obj_dist_c_co_red: Number,
    pub obj_water_consumption_fus: Number,
    pub obj_water_consumption_red: Number,
    pub obj_stove_energy_fus: Number,
    pub obj_stove_energy_red: Number,
    pub obj_post_comb_energy_fus: Number,
    pub obj_post_comb_energy_red: Number,

    pub no_convergence: bool,
    pub error_matter_fus: f64,
    pub error_matter_red: f64,
    pub error_energy_fus: f64,
    pub error_energy_red: f64,
}

impl OptimizationHelpers for JSONManager {}

impl JSONManager {
    pub async fn from_str(data: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let input_data: TheoreticalInput = serde_json::from_str(data)?;
        Self::new(input_data).await
    }
    pub async fn new(input_data: TheoreticalInput) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Self::debug_message("JSON CREATION START".to_string());

        let df_cama = Self::get_cama_data(&input_data)?;
        let df_mineralogy_limestone = Self::get_df_config("calizas_mineralogia", &input_data)?;
        let df_chemistry_limestone = Self::get_df_config("calizas_analisis_quimico", &input_data)?;
        let df_limestone_weight = Self::get_df_config("peso_calizas", &input_data)?;
        let df_recirculated_mineralogy =
            Self::get_df_config("recirculantes_mineralogia", &input_data)?;
        let df_recirculated_chemistry =
            Self::get_df_config("recirculantes_analisis_quimico", &input_data)?;
        let df_carbon_feed = Self::get_df_config("carbon_alimento", &input_data)?;
        let df_fuel_parameters = Self::get_df_config("combustible_parametros", &input_data)?;
        let df_heating = Self::get_df_config("calentamiento_splating", &input_data)?;
        let df_fusion_minor_elements = Self::get_df_config("fusion_minor_elements", &input_data)?;
        let df_reduction_minor_elements =
            Self::get_df_config("reduction_minor_elements", &input_data)?;
        let df_energy_elements = Self::get_df_config("energy_elements", &input_data)?;
        let df_energy_equations = Self::get_df_config("energy_equations", &input_data)?;
        let df_energy_water = Self::get_df_config("energy_water", &input_data)?;

        let hashmap_cama = Self::add_to_hashmap(&df_cama)?;

        let mut sum_recirculated = 0.0;
        let mut sum_prod_recirculated = 0.0;

        for array in input_data.recirculantes.values() {
            sum_recirculated += array[0]; // Suma del primer elemento de cada array
            sum_prod_recirculated += array[0] * array[1]; // Suma producto de los elementos de cada array
        }

        sum_prod_recirculated /= sum_recirculated;
        let obj_fus_stoichiometry = input_data.estequiometria_fusion;
        let obj_red_stoichiometry = input_data.estequiometria_reduccion;
        let obj_o2_fus_enrichment = input_data.enriquecimiento_o2_fusion;
        let obj_o2_red_enrichment = input_data.enriquecimiento_o2_reduccion;

        let fusion_time = input_data.tiempo_fusion;
        let reduction_time = input_data.tiempo_reduccion;

        let air_post_combustion_fus = input_data.aire_post_combustion_fusion;
        let air_infiltration_fus = input_data.aire_infiltracion_fusion;
        let total_air_fus = air_infiltration_fus + air_post_combustion_fus;
        let air_post_combustion_fus_m3 = air_post_combustion_fus * fusion_time;
        let air_infiltration_fus_m3 = air_infiltration_fus * fusion_time;
        let total_air_fus_m3 = air_post_combustion_fus_m3 + air_infiltration_fus_m3;

        let air_post_combustion_red = input_data.aire_post_combustion_reduccion;
        let air_infiltration_red = input_data.aire_infiltracion_reduccion;
        let total_air_red = air_infiltration_red + air_post_combustion_red;
        let air_post_combustion_red_m3 = air_post_combustion_red * reduction_time;
        let air_infiltration_red_m3 = air_infiltration_red * reduction_time;
        let total_air_red_m3 = air_post_combustion_red_m3 + air_infiltration_red_m3;

        // Air
        let air_humidity = input_data.otras_configuraciones["humedad_aire"];
        let air_o2_fus =
            (0.21 * (100. - air_humidity) / 100. * total_air_fus_m3) * 32. / 22.4 / 1000.;
        let air_n2_fus = (0.21 * (100. - air_humidity) / 100. * 79. / 21. * total_air_fus_m3) * 28.
            / 22.4
            / 1000.;
        let air_h2o_fus = air_humidity / 100. * total_air_fus_m3 * 18. / 22.4 / 1000.;

        let air_o2_red =
            (0.21 * (100. - air_humidity) / 100. * total_air_red_m3) * 32. / 22.4 / 1000.;
        let air_n2_red = (0.21 * (100. - air_humidity) / 100. * 79. / 21. * total_air_red_m3) * 28.
            / 22.4
            / 1000.;
        let air_h2o_red = air_humidity / 100. * total_air_red_m3 * 18. / 22.4 / 1000.;

        // Atomized air
        let air_gas_cooler_fus = input_data.otras_configuraciones["aire_lanza_gas_cooler_fus"];
        let air_o2_others_fus =
            (0.21 * (100. - air_humidity) / 100. * air_gas_cooler_fus) * 32. / 22.4 / 1000.
                * fusion_time;
        let air_n2_others_fus =
            (0.21 * (100. - air_humidity) / 100. * 79. / 21. * air_gas_cooler_fus) * 28.
                / 22.4
                / 1000.
                * fusion_time;
        let air_h2o_others_fus =
            air_humidity / 100. * air_gas_cooler_fus * 18. / 22.4 / 1000. * fusion_time;

        let air_gas_cooler_red = input_data.otras_configuraciones["aire_lanza_gas_cooler_red"];
        let air_o2_others_red =
            (0.21 * (100. - air_humidity) / 100. * air_gas_cooler_red) * 32. / 22.4 / 1000.
                * reduction_time;
        let air_n2_others_red =
            (0.21 * (100. - air_humidity) / 100. * 79. / 21. * air_gas_cooler_red) * 28.
                / 22.4
                / 1000.
                * reduction_time;
        let air_h2o_others_red =
            air_humidity / 100. * air_gas_cooler_red * 18. / 22.4 / 1000. * reduction_time;

        let sn_dump_end_fusion = input_data.perc_escoria_fusion;
        let sn_dump_end_reduction = input_data.perc_escoria_reduccion;
        let initial_weight_dump = input_data.diametro_interno.pow(2) * (PI as Number)
            / (4.0 as Number)
            * input_data.nivel_escoria_fusion
            / (1000.0 as Number)
            * input_data.densidad_escoria;

        Self::debug_message("JSON CREATION END".to_string());
        Ok(JSONManager {
            theoretical_input: input_data,
            hashmap_cama,
            df_cama,
            df_mineralogy_limestone,
            df_chemistry_limestone,
            df_limestone_weight,
            df_recirculated_mineralogy,
            df_recirculated_chemistry,
            df_carbon_feed,
            df_fuel_parameters,
            df_heating,
            df_fusion_minor_elements,
            df_reduction_minor_elements,
            df_energy_elements,
            df_energy_equations,
            df_energy_water,
            recirculated_tmh: sum_recirculated,
            recirculated_humidity: sum_prod_recirculated,
            initial_weight_dump,
            free_oxygen_fus: -1.,
            free_oxygen_red: -1.,
            fuel_factor: -1.,
            fuel_heating: -1.,
            time_heating: -1.,
            heating_stoichiometry: -1.,
            c_air_fus: -1.,
            metal_drained: -1.,

            sn_dump_end_fusion,
            sn_dump_end_reduction,
            fusion_time,
            reduction_time,

            air_post_combustion_fus,
            air_infiltration_fus,
            total_air_fus,
            total_air_red,
            air_post_combustion_fus_m3,
            air_infiltration_fus_m3,
            air_post_combustion_red_m3,
            air_infiltration_red_m3,
            total_air_fus_m3,
            total_air_red_m3,

            air_post_combustion_red,
            air_infiltration_red,

            air_humidity,
            air_o2_fus,
            air_n2_fus,
            air_h2o_fus,
            air_o2_red,
            air_n2_red,
            air_h2o_red,
            air_o2_others_fus,
            air_n2_others_fus,
            air_h2o_others_fus,
            air_o2_others_red,
            air_n2_others_red,
            air_h2o_others_red,
            total_dist_tm_fe: -1.,
            free_c_fus: 1.,
            free_c_red: 1.,

            co_o5_fus: -1.,
            co_o5_red: -1.,
            obj_carbon_fus: 1.0,
            obj_carbon_red: 1.0,
            obj_gas_fus: 1600.,
            obj_gas_red: 1600.,
            obj_fus_stoichiometry,
            obj_red_stoichiometry,
            obj_o2_fus_enrichment,
            obj_o2_red_enrichment,
            obj_air_result_nm3_h_fus: -1.,
            obj_oxygen_result_nm3_h_fus: -1.,
            obj_air_result_nm3_h_red: -1.,
            obj_oxygen_result_nm3_h_red: -1.,
            obj_distribution_sno_dump_fus: 1.0,
            obj_distribution_sno_dump_red: 1.0,
            obj_dist_feo_fe_fus: 1.0,
            obj_dist_feo_fe_red: 1.0,
            obj_dist_c_co_fus: 45.2,
            obj_dist_c_co_red: 50.0,
            obj_water_consumption_fus: 19000.,
            obj_water_consumption_red: 19000.,
            obj_stove_energy_fus: 1.0,
            obj_post_comb_energy_fus: 1.0,
            obj_stove_energy_red: 1.0,
            obj_post_comb_energy_red: 1.0,

            no_convergence: false,
            error_matter_fus: 0.0,
            error_energy_fus: 0.0,
            error_matter_red: 0.0,
            error_energy_red: 0.0,
        })
    }

    fn get_df_config(
        config_name: &str,
        input_data: &TheoreticalInput,
    ) -> Result<DataFrame, Box<dyn Error + Send + Sync>> {
        let content: String = input_data.dataframes[config_name].clone();
        let cursor = Cursor::new(content);
        let mut df = JsonReader::new(cursor).finish()?;
        df.cast_numeric();
        debug!(
            "**************** {:?}: **************** {:?}",
            config_name, df
        );
        Ok(df)
    }

    fn get_cama_data(
        input_data: &TheoreticalInput,
    ) -> Result<DataFrame, Box<dyn Error + Send + Sync>> {
        let content: String = input_data.cama_dataframe.clone();
        let cursor = Cursor::new(content);
        let mut df = JsonReader::new(cursor).finish()?;
        df.apply("tmh", |s| s / input_data.num_batches)?;
        df.apply("tms", |s| s / input_data.num_batches)?;
        df.cast_numeric();
        debug!("**************** Cama: **************** {:?}", df);
        Ok(df)
    }

    fn weighted_average(df_cama: &DataFrame, a: &str, b: &str) -> Result<Number, PolarsError> {
        let a_col = df_cama.column(a)?;
        let b_col = df_cama.column(b)?;
        let product = (a_col * b_col).sum::<Number>()?;
        let a_sum = a_col.sum::<Number>()?;
        Ok(product / a_sum)
    }

    fn add_to_hashmap(df_cama: &DataFrame) -> Result<HashMap<String, Number>, PolarsError> {
        let mut results = HashMap::new();

        let total_tmh = df_cama.column("tmh")?.sum::<Number>()?;
        let total_tms = df_cama.column("tms")?.sum::<Number>()?;
        results.insert("tmh".to_string(), total_tmh);
        results.insert("tms".to_string(), total_tms);

        let metrics = [
            ("h2o", "h2o"),
            ("sio2", "sio2"),
            ("s", "s"),
            ("pb", "pb"),
            ("sn", "sn"),
            ("sb", "sb"),
            ("as_field", "as"),
            ("fe", "fe"),
            ("al2o3", "al2o3"),
            ("cu", "cu"),
            ("ca", "cao"),
        ];

        for (column, key) in metrics.iter() {
            if column == &"h2o" {
                let value = Self::weighted_average(df_cama, "tmh", column)?;
                results.insert(key.to_string(), value);
            } else {
                let value = Self::weighted_average(df_cama, "tms", column)?;
                results.insert(key.to_string(), value);
            }
        }

        let min_sno2 = results["sn"] * mw("sno2") / mw("sn");
        let min_fes2 = results["s"] * mw("fes2") / mw("s2");
        let min_fe2o3 =
            (results["fe"] - min_fes2 * mw("fe") / mw("fes2")) * mw("fe2o3") / mw("fe2");

        results.insert("sno2".to_string(), min_sno2);
        results.insert("fes2".to_string(), min_fes2);
        results.insert("fe2o3".to_string(), min_fe2o3);

        let others = 100.0
            - (results["sno2"]
                + results["fe2o3"]
                + results["fes2"]
                + results["sio2"]
                + results["al2o3"]
                + results["pb"]
                + results["sb"]
                + results["as"]
                + results["cu"]
                + results["cao"]);

        results.insert("others".to_string(), others);

        Ok(results)
    }
}

impl Debug for JSONManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("\nJson Manager")
            .field("\ndf_cama", &self.df_cama)
            .field("\ntotal_dist_tm_fe", &self.total_dist_tm_fe)
            .field(
                "\nobj_distribution_sno_dump_fus",
                &self.obj_distribution_sno_dump_fus,
            )
            .field("\nobj_dist_c_co_fus", &self.obj_dist_c_co_fus)
            .field("\nobj_dist_feo_fe_fus", &self.obj_dist_feo_fe_fus)
            .field("\nrecirculated_tmh", &self.recirculated_tmh)
            .field("\nrecirculated_humidity", &self.recirculated_humidity)
            .field("\nco_o5_fus", &self.co_o5_fus)
            .field("\nair_o2_others_fus", &self.air_o2_others_fus)
            .field("\nair_n2_fus", &self.air_n2_fus)
            .field("\nfuel_heating", &self.fuel_heating)
            .field("\ninitial_weight_dump", &self.initial_weight_dump)
            .field("\nfuel_factor", &self.fuel_factor)
            .field("\nsn_dump_end_fusion", &self.sn_dump_end_fusion)
            .field("\nfree_oxygen_fus", &self.free_oxygen_fus)
            .field("\nfree_c_red", &self.free_c_red)
            .field("\nfree_oxygen_red", &self.free_oxygen_red)
            .field("\nobj_air_result_nm3_h_fus", &self.obj_air_result_nm3_h_fus)
            .field(
                "\nobj_oxygen_result_nm3_h_fus",
                &self.obj_oxygen_result_nm3_h_fus,
            )
            .finish()
    }
}
