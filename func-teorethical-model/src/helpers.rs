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

use crate::json_structs::mw;
use crate::Number;
use log::debug;
use polars::error::PolarsError;
use polars::prelude::*;
use std::collections::HashMap;
use std::error::Error;

pub trait OptimizationHelpers {
    fn external_sum_prd(column_a: &Series, column_b: &Series) -> Result<Number, PolarsError> {
        if column_a.len() != column_b.len() {
            return Err(PolarsError::InvalidOperation(
                "Columns length not match for sum_prd".into(),
            ));
        }

        #[cfg(feature = "use_f32")]
        {
            let a: Vec<f32> = column_a.f32()?.into_no_null_iter().collect();
            let b: Vec<f32> = column_b.f32()?.into_no_null_iter().collect();
            let sum_product: f32 = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum();
            Ok(sum_product as Number)
        }

        #[cfg(feature = "use_f64")]
        {
            let a: Vec<f64> = column_a.f64()?.into_no_null_iter().collect();
            let b: Vec<f64> = column_b.f64()?.into_no_null_iter().collect();
            let sum_product: Number = a.iter().zip(b.iter()).map(|(&a, &b)| a * b).sum();
            Ok(sum_product)
        }
    }

    fn create_calories_table(
        df: &DataFrame,
        sense_heat_elements: &Vec<&str>,
        temperature: Number,
    ) -> Result<HashMap<String, Number>, Box<dyn Error>> {
        // Convertimos `sense_heat_elements` a una Serie de Polars sin clonarla
        let elements_series = Series::new("elements", sense_heat_elements);

        // Filtramos el DataFrame usando una referencia al vector en lugar de clonarlo
        let filtered_df = df
            .clone()
            .lazy()
            .filter(col("element").is_in(lit(elements_series)));

        let result_cal_mol = col("a") * (lit(temperature) - col("temp. (ºk)"))
            + col("b") * lit(1e-3) * (lit(temperature).pow(2.0) - col("temp. (ºk)").pow(2.0))
                / lit(2.0)
            - col("c") * lit(1e5) * (lit(1.0) / lit(temperature) - lit(1.0) / col("temp. (ºk)"))
            + col("to cal/mol");

        // Definimos el tipo de columna basado en la configuración
        let lf = filtered_df.with_column(result_cal_mol.alias("cal_mol"));
        let final_result_df = lf
            .with_column((col("cal_mol") / col("pm")).alias("kcal_kg"))
            .collect()?;

        // Convertimos el DataFrame final a un HashMap
        let mut hm = HashMap::new();
        let elements = final_result_df.column("element")?.str()?;
        let kcal_kg = final_result_df.column("kcal_kg")?.f64()?;

        for (opt_element, opt_kcal) in elements.into_iter().zip(kcal_kg.into_iter()) {
            if let (Some(element), Some(kcal)) = (opt_element, opt_kcal) {
                hm.insert(element.to_string(), kcal as Number);
            }
        }

        // Insertamos el valor específico para "h2o_e" si está en `sense_heat_elements`
        if sense_heat_elements.contains(&"h2o_e") {
            hm.insert("h2o_e".to_string(), 1352.25 * temperature / 373. / 18.);
        }

        Ok(hm)
    }

    fn debug_message(msg: String) {
        let border = "*".repeat(50);
        debug!("{} {} {}", border, msg, border);
    }

    fn calculate_masses(
        principal_component_mass: Number,
        left_names: &[&str],
        right_names: &[&str],
    ) -> Result<HashMap<String, Number>, Box<dyn Error>> {
        let principal_molecular_mass = mw(left_names[0]);
        let mut final_equation = HashMap::new();

        // Function to calculate mass for a list of compounds
        fn calculate_mass_for_side(
            names: &[&str],
            principal_component_mass: Number,
            principal_molecular_mass: Number,
            final_equation: &mut HashMap<String, Number>,
        ) -> Number {
            let mut mass_sum = 0.0;
            for compound in names {
                let mass = principal_component_mass * mw(compound) / principal_molecular_mass;
                final_equation.insert(compound.to_string(), mass);
                mass_sum += mass;
            }
            mass_sum
        }

        let left_mass_sum = calculate_mass_for_side(
            left_names,
            principal_component_mass,
            principal_molecular_mass,
            &mut final_equation,
        );
        let right_mass_sum = calculate_mass_for_side(
            right_names,
            principal_component_mass,
            principal_molecular_mass,
            &mut final_equation,
        );

        // Determine if the equation is balanced
        let balance = ((left_mass_sum - right_mass_sum) * 100.).round() / 100.;
        final_equation.insert("balance".to_string(), balance);
        final_equation.insert(
            "equilibrium".to_string(),
            if balance == 0.0 { 1.0 } else { 0.0 },
        );

        Ok(final_equation)
    }
}

pub trait DataFrameCustomExtensions {
    fn get_conditional_value(
        &self,
        column_a: &str,
        column_a_condition: &str,
        column_b: &str,
    ) -> Result<Number, PolarsError>;
    fn get_specific_value(&self, column_a: &str, index: usize) -> Result<Number, PolarsError>;
    fn sum_prd(&self, column_a: &str, column_b: &str) -> Result<Number, PolarsError>;
    fn cast_numeric(&mut self);
    fn generate_hashmap_str_number(
        &self,
        key_col_name: &str,
        value_col_name: &str,
    ) -> Result<HashMap<String, Number>, Box<dyn Error>>;
}

impl DataFrameCustomExtensions for DataFrame {
    fn get_conditional_value(
        &self,
        column_a: &str,
        column_a_condition: &str,
        column_b: &str,
    ) -> Result<Number, PolarsError> {
        let column_a_series = self.column(column_a)?;
        let mask = column_a_series.str()?.equal(column_a_condition);

        // Separate the scope to ensure `mask` is dropped before we continue.
        let idx_opt = {
            let mut mask_iter = mask.into_iter();
            mask_iter.position(|x| x == Some(true))
        };

        if let Some(idx) = idx_opt {
            let column_b_series = self.column(column_b)?;
            #[cfg(feature = "use_f32")]
            let result = column_b_series.f32()?.get(idx).ok_or(PolarsError::NoData(
                "Can't get conditional value from dataframe".into(),
            ))?;

            #[cfg(feature = "use_f64")]
            let result = column_b_series.f64()?.get(idx).ok_or(PolarsError::NoData(
                "Can't get conditional value from dataframe".into(),
            ))?;

            Ok(result)
        } else {
            Err(PolarsError::NoData(
                "Can't get conditional value from dataframe".into(),
            ))
        }
    }

    fn get_specific_value(&self, column_a: &str, index: usize) -> Result<Number, PolarsError> {
        let series = self.column(column_a)?;

        #[cfg(feature = "use_f32")]
        let val = series.f32()?.get(index).ok_or(PolarsError::NoData(
            "Can't get specific value from dataframe".into(),
        ))?;

        #[cfg(feature = "use_f64")]
        let val = series.f64()?.get(index).ok_or(PolarsError::NoData(
            "Can't get specific value from dataframe".into(),
        ))?;

        Ok(val)
    }

    fn sum_prd(&self, column_a: &str, column_b: &str) -> Result<Number, PolarsError> {
        // Clone the DataFrame and calculate the sum of products using lazy evaluation

        let sum_prd_limestone = self
            .clone()
            .lazy()
            .select([(col(column_a) * col(column_b)).sum().alias("sum_prd")])
            .collect()?;
        // Retrieve the computed sum of products and convert it to Number
        let initial_result = sum_prd_limestone.column("sum_prd")?;
        #[cfg(feature = "use_f32")]
        let result = initial_result
            .f32()?
            .get(0)
            .ok_or(PolarsError::NoData("Cannot compute sum product".into()))?;
        #[cfg(feature = "use_f64")]
        let result = initial_result
            .f64()?
            .get(0)
            .ok_or(PolarsError::NoData("Cannot compute sum product".into()))?;

        Ok(result)
    }

    fn cast_numeric(&mut self) {
        // Attempt to convert each column to Number if possible
        for name in self.get_column_names_owned() {
            if let Ok(series) = self.column(&name) {
                if series.dtype() == &DataType::Float64
                    || series.dtype() == &DataType::Int32
                    || series.dtype() == &DataType::Int64
                    || series.dtype() == &DataType::Float32
                {
                    // Try to convert to Float32 and replace the column if successful
                    #[cfg(feature = "use_f32")]
                    if let Ok(mut converted) = series.cast(&DataType::Float32) {
                        self.with_column(converted.rename(&name).clone())
                            .expect("Can't cast to f32");
                    }
                    #[cfg(feature = "use_f64")]
                    if let Ok(mut converted) = series.cast(&DataType::Float64) {
                        self.with_column(converted.rename(&name).clone())
                            .expect("Can't cast to f64");
                    }
                }
            }
        }
    }

    fn generate_hashmap_str_number(
        &self,
        key_col_name: &str,
        value_col_name: &str,
    ) -> Result<HashMap<String, Number>, Box<dyn Error>> {
        // Attempt to retrieve the key column and value column
        let key_col = self.column(key_col_name)?;
        let value_col = self.column(value_col_name)?;

        // Ensure the columns can be converted to type &str or String
        let keys = key_col.str()?;

        #[cfg(feature = "use_f32")]
        let values = value_col.f32()?;

        #[cfg(feature = "use_f64")]
        let values = value_col.f64()?;

        let mut map = HashMap::with_capacity(keys.len());

        for (opt_key, opt_value) in keys.into_iter().zip(values.into_iter()) {
            if let (Some(key), Some(value)) = (opt_key, opt_value) {
                map.insert(key.to_owned(), value as Number);
            }
        }

        Ok(map)
    }
}
