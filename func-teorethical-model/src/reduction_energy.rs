use crate::fusion_energy::FusionEnergy;
use crate::fusion_matter::FusionMatter;
use crate::helpers::{DataFrameCustomExtensions, OptimizationHelpers};
use crate::json_manager::JSONManager;
use crate::reduction_matter::ReductionMatter;
use crate::Number;
use std::collections::HashMap;
use std::error::Error;

#[derive(Debug)]
pub struct ReductionEnergy {
    pub total_heat_input: Number,
    pub hm_heat_input: HashMap<String, Number>,
    pub total_heat_output: Number,
    pub hm_total_heat_output: HashMap<String, Number>,
    pub total_input_heat_post_combustion: Number,
    pub hm_input_heat_post_combustion: HashMap<String, Number>,
    pub total_output_heat_post_combustion: Number,
    pub hm_output_heat_post_combustion: HashMap<String, Number>,
    pub total_input_heat_gas_cooler: Number,
    pub total_output_heat_gas_cooler: Number,
}

impl OptimizationHelpers for ReductionEnergy {}

impl ReductionEnergy {
    pub fn new(
        json_manager: &mut JSONManager,
        fusion_matter: &FusionMatter,
        reduction_matter: &ReductionMatter,
        fusion_energy: &FusionEnergy,
    ) -> Result<Self, Box<dyn Error>> {
        let (total_heat_input, hm_heat_input) = Self::create_total_heat_input(
            json_manager,
            fusion_matter,
            reduction_matter,
            fusion_energy,
        )?;

        let (total_heat_output, hm_total_heat_output) =
            Self::create_total_heat_output(json_manager, reduction_matter)?;

        let (total_input_heat_post_combustion, hm_input_heat_post_combustion) =
            Self::create_total_input_heat_post_combustion(
                json_manager,
                reduction_matter,
                &hm_total_heat_output,
            )?;

        let (total_output_heat_post_combustion, hm_output_heat_post_combustion) =
            Self::create_total_output_heat_post_combustion(
                json_manager,
                reduction_matter,
                total_input_heat_post_combustion,
            )?;

        let (total_input_heat_gas_cooler, total_output_heat_gas_cooler) =
            Self::create_total_heat_gas_cooler(
                json_manager,
                reduction_matter,
                hm_output_heat_post_combustion["mcal_gases_heat_sense"],
                hm_output_heat_post_combustion["mcal_fumes_heat_sense"],
            )?;

        Ok(ReductionEnergy {
            total_heat_input,
            hm_heat_input,
            total_heat_output,
            hm_total_heat_output,
            total_input_heat_post_combustion,
            hm_input_heat_post_combustion,
            total_output_heat_post_combustion,
            hm_output_heat_post_combustion,
            total_input_heat_gas_cooler,
            total_output_heat_gas_cooler,
        })
    }

    fn create_total_heat_input(
        json_manager: &JSONManager,
        fusion_matter: &FusionMatter,
        reduction_matter: &ReductionMatter,
        fusion_energy: &FusionEnergy,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1. Petroleum Combustion
        let df_combustion_income = &reduction_matter.df_combustion_income;
        let tm_1 =
            df_combustion_income.get_conditional_value("components", "total", "combustible")?;
        let df_f_p = &json_manager.df_fuel_parameters;
        let petroleum_combustion_kcal_kg = df_f_p.get_specific_value("kcal/m3", 0)?;
        let pm = df_f_p.get_specific_value("pm", 0)?;
        let kcal_kg_1 = petroleum_combustion_kcal_kg * 22.4 / pm;
        let mcal_petroleum_combustion = tm_1 * kcal_kg_1;

        // 2. Heat Petroleum Sense
        let temperature_2 = 105.;
        let j_kg_h = 1700.;
        let mcal_heat_sense_petroleum: Number =
            j_kg_h / 4.186 * tm_1 * (temperature_2 + 273.) / 1000.;

        // 3. Heat Sense Initial Metal
        let total_metal = fusion_matter
            .df_metal
            .get_conditional_value("elements", "total", "tm")?;
        let mcal_metal_heat_sense = (total_metal - json_manager.metal_drained * 1.5)
            * fusion_energy.hm_total_heat_output["mcal_metal_heat_sense"]
            / total_metal;

        // 4. Heat Sense Dump
        let mcal_dump_heat_sense = fusion_energy.hm_total_heat_output["mcal_dump_heat_sense"];

        // 5. Heat Sense Air
        let temperature_5 = 31.0;
        let hm_air_combustion_income =
            df_combustion_income.generate_hashmap_str_number("components", "air")?;
        let hm_a_air_combustion_income =
            df_combustion_income.generate_hashmap_str_number("components", "a_atomized")?;
        let o2_inf = json_manager.air_o2_red;
        let n2_inf = json_manager.air_n2_red;
        let h2o_inf = json_manager.air_h2o_red;
        let lance_reactions_red =
            json_manager.theoretical_input.otras_configuraciones["lanza_reacciona_reduccion"];
        let air_post_combustion_red = json_manager.air_post_combustion_red;
        let air_infiltration_red = json_manager.air_infiltration_red;
        let o2_4_tm = hm_air_combustion_income["o"]
            + hm_a_air_combustion_income["o"]
            + o2_inf * air_post_combustion_red / air_infiltration_red * lance_reactions_red;
        let n2_4_tm = hm_air_combustion_income["n"]
            + hm_a_air_combustion_income["n"]
            + n2_inf * air_post_combustion_red / air_infiltration_red * lance_reactions_red;
        let h2o_4_tm = hm_air_combustion_income["h2o"]
            + hm_a_air_combustion_income["h2o"]
            + h2o_inf * air_post_combustion_red / air_infiltration_red * lance_reactions_red;
        let sense_air_elements = vec!["o2", "n2", "h2o_e"];

        let mut tm_sense_air = HashMap::new();
        tm_sense_air.insert("o2", o2_4_tm);
        tm_sense_air.insert("n2", n2_4_tm);
        tm_sense_air.insert("h2o_e", h2o_4_tm);
        let hm_sense_air_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &sense_air_elements,
            temperature_5 + 273.,
        )?;
        let mcal_sense_air: Number = sense_air_elements
            .iter()
            .map(|&element| hm_sense_air_calories[element] * 1000. * tm_sense_air[element])
            .sum::<Number>()
            / 1000.;

        // 6. Heat Sense Oxygen
        let temperature_6: Number = 27.;
        let o2_5_tm = df_combustion_income.get_conditional_value("components", "o", "oxygen")?;
        let n2_5_tm = df_combustion_income.get_conditional_value("components", "n", "oxygen")?;
        let h2o_5_tm = df_combustion_income.get_conditional_value("components", "h2o", "oxygen")?;
        let sense_oxygen_elements = vec!["o2", "n2", "h2o_e"];
        let hm_sense_oxygen_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &sense_oxygen_elements,
            temperature_6 + 273.,
        )?;

        let mut tm_sense_oxygen = HashMap::new();
        tm_sense_oxygen.insert("o2", o2_5_tm);
        tm_sense_oxygen.insert("n2", n2_5_tm);
        tm_sense_oxygen.insert("h2o_e", h2o_5_tm);

        let mcal_sense_oxygen: Number = sense_oxygen_elements
            .iter()
            .map(|&element| hm_sense_oxygen_calories[element] * 1000. * tm_sense_oxygen[element])
            .sum::<Number>()
            / 1000.;

        // Chemical exo reactions
        let elements = [
            "cu2s + sn = sns + 2 cu",
            "s + sn = sns",
            "c2h6 + 3.5 o2 = 2 co2 + 3 h2o",
            "co + 0.5 o2 = co2",
            "c + 0.5 o2  = co",
            "sno + 0.5 o2 = sno2",
            "sns + 2 o2 = sno2 + so2",
            "0.5 s2 + o2 = so2",
            "c2h6 + 3.5 o2 = 2 co2 + 3 h2o",
        ];

        let reactions = &reduction_matter.reactions;
        let hm_reactions_kcal = json_manager
            .df_energy_equations
            .generate_hashmap_str_number("element", "kcal/kg")?;
        let mut hm_reactions_1 = HashMap::new();
        let mut hm_reactions_2 = HashMap::new();

        hm_reactions_1.insert("cu2s + sn = sns + 2 cu", reactions[12]["cu2s"]);
        hm_reactions_1.insert("s + sn = sns", reactions[13]["s"]);
        hm_reactions_1.insert("c2h6 + 3.5 o2 = 2 co2 + 3 h2o", reactions[14]["c2h6"]);
        hm_reactions_1.insert("co + 0.5 o2 = co2", reactions[15]["co"]);
        hm_reactions_1.insert("c + 0.5 o2  = co", reactions[16]["c"]);

        hm_reactions_2.insert("sno + 0.5 o2 = sno2", reactions[22]["sno"]);
        hm_reactions_2.insert("sns + 2 o2 = sno2 + so2", reactions[23]["sns"]);
        hm_reactions_2.insert("co + 0.5 o2 = co2", reactions[24]["co"]);
        hm_reactions_2.insert("0.5 s2 + o2 = so2", reactions[25]["0.5s2"]);
        hm_reactions_2.insert("c2h6 + 3.5 o2 = 2 co2 + 3 h2o", reactions[26]["c2h6"]);

        let mcal_reactions: Number = elements
            .iter()
            .map(|&element| {
                let r_1 = hm_reactions_1.get(element).unwrap_or(&0.0)
                    * hm_reactions_kcal[element]
                    * -1000.;
                let r_2 = hm_reactions_2.get(element).unwrap_or(&0.0)
                    * hm_reactions_kcal[element]
                    * -1000.;
                r_1 + r_2
            })
            .sum::<Number>()
            / 1000.;

        let result = mcal_reactions
            + mcal_sense_oxygen
            + mcal_sense_air
            + mcal_dump_heat_sense
            + mcal_heat_sense_petroleum
            + mcal_metal_heat_sense
            + mcal_petroleum_combustion;

        let mut hm_heat_input = HashMap::new();
        hm_heat_input.insert("mcal_reactions".to_string(), mcal_reactions);
        hm_heat_input.insert("mcal_sense_oxygen".to_string(), mcal_sense_oxygen);
        hm_heat_input.insert("mcal_sense_air".to_string(), mcal_sense_air);
        hm_heat_input.insert("mcal_dump_heat_sense".to_string(), mcal_dump_heat_sense);
        hm_heat_input.insert("mcal_metal_heat_sense".to_string(), mcal_metal_heat_sense);
        hm_heat_input.insert(
            "mcal_petroleum_combustion".to_string(),
            mcal_petroleum_combustion,
        );
        hm_heat_input.insert(
            "mcal_heat_sense_petroleum".to_string(),
            mcal_heat_sense_petroleum,
        );

        Ok((result, hm_heat_input))
    }

    fn create_total_heat_output(
        json_manager: &JSONManager,
        reduction_matter: &ReductionMatter,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1. Metal
        let obj_stove_energy_reduction = json_manager.obj_stove_energy_red;
        let hm_metal = reduction_matter
            .df_metal
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_1 = vec!["sn", "fe", "pb", "sb", "as", "cu", "otros"];

        let hm_1_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_1,
            obj_stove_energy_reduction + 273. - 200. - 25.,
        )?;
        let mut mcal_metal_heat_sense = 0.;
        for element in elements_1 {
            if element == "otros" {
                mcal_metal_heat_sense += hm_metal["others"] * 1000. * hm_1_calories["otros"];
            } else {
                mcal_metal_heat_sense += hm_metal[element] * 1000. * hm_1_calories[element];
            }
        }
        mcal_metal_heat_sense /= 1000.;

        // 2. Dump
        let hm_dump = reduction_matter
            .df_dump
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_2 = vec!["sno", "feo", "sio2", "al2o3", "cao", "mgo", "otros"];
        let hm_2_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_2,
            obj_stove_energy_reduction + 273.,
        )?;
        let mut mcal_dump_heat_sense = 0.;
        for element in elements_2 {
            if element == "otros" {
                mcal_dump_heat_sense += hm_dump["others"] * 1000. * hm_2_calories["otros"];
            } else {
                mcal_dump_heat_sense += hm_dump[element] * 1000. * hm_2_calories[element];
            }
        }

        mcal_dump_heat_sense /= 1000.;

        // 3. Gas
        let hm_gases = &reduction_matter.hm_gases;
        let elements_3 = vec!["co2", "co", "n2", "so2", "h2o", "ch4", "o2", "s2"];
        let hm_3_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_3,
            obj_stove_energy_reduction + 50. + 273.,
        )?;

        let mut mcal_gases_heat_sense = 0.;
        for element in elements_3 {
            if element == "ch4" {
                mcal_gases_heat_sense += hm_gases["c2h6"] * 1000. * hm_3_calories["ch4"];
            } else {
                mcal_gases_heat_sense += hm_gases[element] * 1000. * hm_3_calories[element];
            }
        }

        mcal_gases_heat_sense /= 1000.;

        // 4. Dross Fe
        let hm_dross = reduction_matter
            .df_dross_fe
            .generate_hashmap_str_number("elements", "tm")?;

        let elements_4 = vec!["sn", "fe", "pb", "sb", "as", "cu", "otros"];
        let hm_4_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_4,
            obj_stove_energy_reduction + 273. - 200. - 25.,
        )?;
        let mut mcal_dross_heat_sense = 0.;
        for element in elements_4 {
            if element == "otros" {
                mcal_dross_heat_sense += hm_dross["others"] * 1000. * hm_4_calories["otros"];
            } else {
                mcal_dross_heat_sense += hm_dross[element] * 1000. * hm_4_calories[element];
            }
        }

        mcal_dross_heat_sense /= 1000.;

        // 5. Fumes
        let hm_fumes = reduction_matter
            .df_fumes
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_5 = vec![
            "sno v", "sno2", "sns v", "feo", "pb", "sb", "as", "cu", "otros",
        ];
        let hm_5_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_5,
            obj_stove_energy_reduction + 50. + 273.,
        )?;
        let mut mcal_fumes_heat_sense = 0.;
        for element in elements_5 {
            if element == "otros" {
                mcal_fumes_heat_sense += hm_fumes["others"] * 1000. * hm_5_calories[element];
            } else if element == "sno v" {
                mcal_fumes_heat_sense += hm_fumes["sno"] * 1000. * hm_5_calories["sno v"];
            } else if element == "sns v" {
                mcal_fumes_heat_sense += hm_fumes["sns"] * 1000. * hm_5_calories["sns v"];
            } else {
                mcal_fumes_heat_sense += hm_fumes[element] * 1000. * hm_5_calories[element];
            }
        }
        mcal_fumes_heat_sense /= 1000.;

        // 6.
        let mut hm_reactions_6 = HashMap::new();
        let reactions = &reduction_matter.reactions;
        let reactions_calories = &json_manager
            .df_energy_equations
            .generate_hashmap_str_number("element", "kcal/kg")?;
        let reactions_list = vec![
            "sno2 + c = sno + co",
            "fes2 + sno = sns + feo + 0.5 s2",
            "fesn2   = 2 sn + fe",
            "fe + sno = feo + sn",
            "sno + c = sn + co",
            "fe2o3 + c = 2 feo + co",
            "feo + c = fe + co",
            "caco3  = cao + co2",
            "mgco3  = mgo + co2",
        ];
        hm_reactions_6.insert("sno2 + c = sno + co", reactions[0]["sno2"]);
        hm_reactions_6.insert("fes2 + sno = sns + feo + 0.5 s2", reactions[1]["fes2"]);
        hm_reactions_6.insert("fesn2   = 2 sn + fe", reactions[2]["fesn2"]);
        hm_reactions_6.insert("fe + sno = feo + sn", reactions[3]["fe"]);
        hm_reactions_6.insert("sno + c = sn + co", reactions[4]["sno"]);
        hm_reactions_6.insert("fe2o3 + c = 2 feo + co", reactions[7]["fe2o3"]);
        hm_reactions_6.insert("feo + c = fe + co", reactions[8]["feo"]);
        hm_reactions_6.insert("caco3  = cao + co2", reactions[10]["caco3"]);
        hm_reactions_6.insert("mgco3  = mgo + co2", reactions[11]["mgco3"]);

        let mut mcal_reactions = 0.;
        for reaction in reactions_list {
            mcal_reactions += hm_reactions_6[reaction] * 1000. * reactions_calories[reaction]
        }
        mcal_reactions /= 1000.;

        // 7. Latent Heat
        let hm_income_charge = reduction_matter
            .df_income_charge
            .generate_hashmap_str_number("molecules", "total")?;
        let hm_water_calories = &json_manager
            .df_energy_water
            .generate_hashmap_str_number("element", "kcal/kg")?;

        let water_7_tm = hm_income_charge["h2o"];
        let sn_7_tm = hm_income_charge["sn"] + reactions[2]["2sn"];
        let sno_7_tm = reactions[5]["sno(v)"];
        let sns_7_tm = reduction_matter
            .df_fumes
            .get_conditional_value("elements", "sns", "tm")?;

        let mcal_latent_heat: Number = water_7_tm * hm_water_calories["evaporacion agua"]
            + sn_7_tm * hm_water_calories["sn(s) = sn(l)"]
            + sno_7_tm * hm_water_calories["sno(s) = sno(g)"]
            + sns_7_tm * hm_water_calories["sns(s) = sns(g)"];

        // 8. Refrigeration heat

        let water_flow_8 = json_manager.theoretical_input.otras_configuraciones["flujo_agua"];
        let initial_temp_8 = json_manager
            .theoretical_input
            .temperatura_inicial_reduccion_horno;
        let final_temp_8 = json_manager
            .theoretical_input
            .temperatura_final_reduccion_horno;

        let mcal_refrigeration = water_flow_8 * 1000. * (final_temp_8 - initial_temp_8) / 1000.
            * json_manager.reduction_time;
        let lost = json_manager.theoretical_input.otras_configuraciones["perdidas_reduccion"];
        let mcal_heat_lost = lost * json_manager.reduction_time;

        let mut hm_result = HashMap::new();
        hm_result.insert("mcal_metal_heat_sense".to_string(), mcal_metal_heat_sense);
        hm_result.insert("mcal_dump_heat_sense".to_string(), mcal_dump_heat_sense);
        hm_result.insert("mcal_gases_heat_sense".to_string(), mcal_gases_heat_sense);
        hm_result.insert("mcal_dross_heat_sense".to_string(), mcal_dross_heat_sense);
        hm_result.insert("mcal_fumes_heat_sense".to_string(), mcal_fumes_heat_sense);
        hm_result.insert("mcal_refrigeration".to_string(), mcal_refrigeration);
        hm_result.insert("mcal_reactions".to_string(), mcal_reactions);
        hm_result.insert("mcal_heat_lost".to_string(), mcal_heat_lost);
        hm_result.insert("mcal_latent_heat".to_string(), mcal_latent_heat);

        let result = mcal_metal_heat_sense
            + mcal_dump_heat_sense
            + mcal_gases_heat_sense
            + mcal_dross_heat_sense
            + mcal_fumes_heat_sense
            + mcal_refrigeration
            + mcal_reactions
            + mcal_heat_lost
            + mcal_latent_heat;

        Ok((result, hm_result))
    }

    fn create_total_input_heat_post_combustion(
        json_manager: &JSONManager,
        reduction_matter: &ReductionMatter,
        hm_total_heat_output: &HashMap<String, Number>,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1. Gases Sense Heat
        let mcal_gases_heat_sense = hm_total_heat_output["mcal_gases_heat_sense"];
        // 2. Fumes Sense Heat
        let mcal_fumes_heat_sense = hm_total_heat_output["mcal_fumes_heat_sense"];

        // 3. Air Sense heat
        let temperature_3 = 31.;
        let o2_inf = json_manager.air_o2_red;
        let n2_inf = json_manager.air_n2_red;
        let h2o_inf = json_manager.air_h2o_red;
        let air_post_combustion_red = json_manager.air_post_combustion_red;
        let air_infiltration_red = json_manager.air_infiltration_red;
        let code_reaction_fus =
            json_manager.theoretical_input.otras_configuraciones["codo_reacciona_reduccion"];
        let elements = vec!["o2", "n2", "h2o_e"];
        let hm_air_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements,
            temperature_3 + 273.,
        )?;

        let ratio = air_post_combustion_red / (air_infiltration_red + air_post_combustion_red);
        let o2_3_tm = o2_inf * ratio * code_reaction_fus;
        let n2_3_tm = n2_inf * ratio * code_reaction_fus;
        let h2o_3_tm = h2o_inf * ratio * code_reaction_fus;

        let mcal_air_heat_sense = o2_3_tm * hm_air_calories["o2"]
            + h2o_3_tm * hm_air_calories["h2o_e"]
            + n2_3_tm * hm_air_calories["n2"];

        // 4. Reactions Heat Sense
        let hm_energy_equations = json_manager
            .df_energy_equations
            .generate_hashmap_str_number("element", "kcal/kg")?;
        let equations_4 = [
            ("sno + 0.5 o2 = sno2", 27, "sno"),
            ("sns + 2 o2 = sno2 + so2", 28, "sns"),
            ("co + 0.5 o2 = co2", 29, "co"),
            ("0.5 s2 + o2 = so2", 30, "0.5s2"),
            ("c2h6 + 3.5 o2 = 2 co2 + 3 h2o", 31, "c2h6"),
        ];

        let mut mcal_reactions_heat_sense: Number = 0.;
        for &(equation, index, element) in &equations_4 {
            mcal_reactions_heat_sense +=
                -reduction_matter.reactions[index][element] * hm_energy_equations[equation];
        }

        let result = mcal_reactions_heat_sense
            + mcal_gases_heat_sense
            + mcal_fumes_heat_sense
            + mcal_air_heat_sense;

        let mut hm_input_heat_post_combustion = HashMap::new();
        hm_input_heat_post_combustion.insert(
            "mcal_reactions_heat_sense".to_string(),
            mcal_reactions_heat_sense,
        );
        hm_input_heat_post_combustion
            .insert("mcal_gases_heat_sense".to_string(), mcal_gases_heat_sense);
        hm_input_heat_post_combustion
            .insert("mcal_fumes_heat_sense".to_string(), mcal_fumes_heat_sense);
        hm_input_heat_post_combustion
            .insert("mcal_air_heat_sense".to_string(), mcal_air_heat_sense);

        Ok((result, hm_input_heat_post_combustion))
    }

    fn create_total_output_heat_post_combustion(
        json_manager: &JSONManager,
        reduction_matter: &ReductionMatter,
        total_input_heat_post_combustion: Number,
    ) -> Result<(Number, HashMap<String, Number>), Box<dyn Error>> {
        // 1. Heat Sense Gases
        let hm_gases_post = &reduction_matter.hm_gases_post;
        let temperature_1 = json_manager.obj_post_comb_energy_red;
        let elements_1 = vec!["co2", "co", "n2", "so2", "h2o", "o2"];
        let hm_1_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_1,
            temperature_1 + 273.,
        )?;

        let mut mcal_gases_heat_sense: Number = 0.;
        for element in elements_1 {
            mcal_gases_heat_sense += hm_gases_post[element] * hm_1_calories[element];
        }

        // 2. Heat Sense Fumes
        let elements_2 = vec!["sno2", "feo", "pb", "sb", "as", "cu", "otros"];

        let hm_2_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_2,
            temperature_1 + 273.,
        )?;
        let hm_post_combustion_fumes = &reduction_matter.hm_post_combustion_fumes;

        let mut mcal_fumes_heat_sense: Number = 0.;
        for element in elements_2 {
            if element == "otros" {
                mcal_fumes_heat_sense +=
                    hm_2_calories[element] * hm_post_combustion_fumes["others"];
            } else {
                mcal_fumes_heat_sense += hm_2_calories[element] * hm_post_combustion_fumes[element];
            }
        }

        // 3. Heat losses
        let lost_1 = json_manager.theoretical_input.otras_configuraciones["post_comb_red"];
        let mcal_loses = total_input_heat_post_combustion * lost_1 / 100.;

        let result = mcal_loses + mcal_fumes_heat_sense + mcal_gases_heat_sense;
        let mut hm_output_heat_post_combustion = HashMap::new();
        hm_output_heat_post_combustion.insert("mcal_loses".to_string(), mcal_loses);
        hm_output_heat_post_combustion
            .insert("mcal_fumes_heat_sense".to_string(), mcal_fumes_heat_sense);
        hm_output_heat_post_combustion
            .insert("mcal_gases_heat_sense".to_string(), mcal_gases_heat_sense);

        Ok((result, hm_output_heat_post_combustion))
    }

    fn create_total_heat_gas_cooler(
        json_manager: &JSONManager,
        reduction_matter: &ReductionMatter,
        mcal_gases_heat_sense: Number,
        mcal_fumes_heat_sense: Number,
    ) -> Result<(Number, Number), Box<dyn Error>> {
        let eq_1 = json_manager.df_energy_equations.get_conditional_value(
            "element",
            "co + 0.5 o2 = co2",
            "kcal/kg",
        )?;
        let mcal_exo_reaction = eq_1 * json_manager.co_o5_red;

        let total_input_heat_gas_cooler =
            mcal_exo_reaction + mcal_fumes_heat_sense + mcal_gases_heat_sense;

        // Fall apart
        // 1. Water evaporation
        let water_consumption_fus = json_manager.obj_water_consumption_red;
        let reduction_time = json_manager.reduction_time;
        let water_evaporation_kcal = json_manager.df_energy_water.get_conditional_value(
            "element",
            "evaporacion agua",
            "kcal/kg",
        )?;
        let mcal_water_evaporation =
            water_consumption_fus * reduction_time / 1000. * water_evaporation_kcal;

        // 2. Heat Sense Gas Cooler
        let hm_gases_chimney = reduction_matter
            .df_gases_chimney
            .generate_hashmap_str_number("elements", "tm")?;
        let elements_2 = vec!["co2", "co", "n2", "so2", "h2o", "o2"];
        let temperature = json_manager
            .theoretical_input
            .temperatura_salida_gas_cooler_reduccion;
        let hm_2_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_2,
            temperature + 273.,
        )?;

        let mut mcal_gases_sense: Number = 0.;
        for element in elements_2 {
            mcal_gases_sense += hm_gases_chimney[element] * hm_2_calories[element];
        }

        // 3. Heat Sense Fumes
        let hm_post_combustion_fumes = &reduction_matter.hm_post_combustion_fumes;
        let elements_3 = vec!["sno2", "feo", "pb", "sb", "as", "cu", "otros"];
        let hm_3_calories = Self::create_calories_table(
            &json_manager.df_energy_elements,
            &elements_3,
            temperature + 273.,
        )?;
        let mut mcal_fumes_sense: Number = 0.;
        for element in elements_3 {
            if element == "otros" {
                mcal_fumes_sense += hm_post_combustion_fumes["others"] * hm_3_calories[element];
            } else {
                mcal_fumes_sense += hm_post_combustion_fumes[element] * hm_3_calories[element];
            }
        }
        let gas_cooler_red = json_manager.theoretical_input.otras_configuraciones["gas_cooler_red"];

        let mcal_lost_heat = gas_cooler_red / 100. * total_input_heat_gas_cooler;

        let total_output_heat_gas_cooler =
            mcal_water_evaporation + mcal_gases_sense + mcal_fumes_sense + mcal_lost_heat;

        Ok((total_input_heat_gas_cooler, total_output_heat_gas_cooler))
    }
}
