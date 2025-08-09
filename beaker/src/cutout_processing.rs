use crate::color_utils::symbols;
use crate::config::CutoutConfig;
use crate::cutout_postprocessing::{
    apply_alpha_matting, create_cutout, create_cutout_with_background, postprocess_mask,
};
use crate::cutout_preprocessing::preprocess_image_for_isnet_v2;
use crate::model_access::{ModelAccess, ModelInfo};
use crate::onnx_session::ModelSource;
use crate::output_manager::OutputManager;
use crate::shared_metadata::IoTiming;
use anyhow::Result;
use image::GenericImageView;
use log::debug;
use ort::{session::Session, value::Value};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// ISNet General Use default model information
pub fn get_default_cutout_model_info() -> ModelInfo {
    ModelInfo {
        name: "isnet-general-use-v1".to_string(),
        url: "https://github.com/ericphanson/beaker/releases/download/beaker-cutout-model-v1/isnet-general-use.onnx".to_string(),
        md5_checksum: "fc16ebd8b0c10d971d3513d564d01e29".to_string(),
        filename: "isnet-general-use.onnx".to_string(),
    }
}

/// Cutout model access implementation.
pub struct CutAccess;

impl ModelAccess for CutAccess {
    fn get_embedded_bytes() -> Option<&'static [u8]> {
        // Cutout models are not embedded, they are downloaded
        None
    }

    fn get_env_var_name() -> &'static str {
        "BEAKER_CUTOUT_MODEL_PATH"
    }

    fn get_url_env_var_name() -> Option<&'static str> {
        Some("BEAKER_CUTOUT_MODEL_URL")
    }

    fn get_checksum_env_var_name() -> Option<&'static str> {
        Some("BEAKER_CUTOUT_MODEL_CHECKSUM")
    }

    fn get_default_model_info() -> Option<ModelInfo> {
        Some(get_default_cutout_model_info())
    }
}

/// Core results for enhanced metadata (without config duplication)
#[derive(Serialize)]
pub struct CutoutResult {
    pub model_version: String,
    #[serde(skip_serializing)]
    pub processing_time_ms: f64,
    pub output_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_path: Option<String>,
    #[serde(skip_serializing)]
    pub io_timing: IoTiming,
    // Store raw mask data for metadata encoding
    #[serde(skip)]
    pub raw_mask_data: Option<(Vec<u8>, u32, u32)>, // (mask_data, width, height)
    /// List of output files actually created (for depfile generation)
    #[serde(skip)]
    pub created_files: Vec<PathBuf>,
}

/// Extract binary mask data from a GrayImage by thresholding at 128
/// Returns (binary_data, width, height)
fn extract_binary_mask_data(mask: &image::GrayImage) -> (Vec<u8>, u32, u32) {
    let (width, height) = mask.dimensions();
    let mut binary_data = Vec::with_capacity((width * height) as usize);

    for y in 0..height {
        for x in 0..width {
            let pixel = mask.get_pixel(x, y);
            // Threshold at 128: >= 128 becomes 1, < 128 becomes 0
            let binary_value = if pixel[0] >= 128 { 1 } else { 0 };
            binary_data.push(binary_value);
        }
    }

    (binary_data, width, height)
}

/// Process multiple images sequentially
pub fn run_cutout_processing(config: CutoutConfig) -> Result<usize> {
    // Use the new generic processing framework
    crate::model_processing::run_model_processing::<CutoutProcessor>(config)
}

// Implementation of ModelProcessor trait for cutout processing
use crate::model_processing::{ModelProcessor, ModelResult};

impl ModelResult for CutoutResult {
    fn processing_time_ms(&self) -> f64 {
        self.processing_time_ms
    }

    fn tool_name(&self) -> &'static str {
        "cutout"
    }

    fn core_results(&self) -> Result<toml::Value> {
        Ok(toml::Value::try_from(self)?)
    }

    fn output_summary(&self) -> String {
        if self.mask_path.is_some() {
            format!("→ {} + mask", self.output_path)
        } else {
            format!("→ {}", self.output_path)
        }
    }

    fn get_io_timing(&self) -> crate::shared_metadata::IoTiming {
        self.io_timing.clone()
    }

    fn get_mask_entry(&self) -> Option<crate::mask_encoding::MaskEntry> {
        if let Some((mask_data, width, height)) = &self.raw_mask_data {
            match crate::mask_encoding::encode_mask_to_entry(mask_data, *width, *height, 0) {
                Ok(entry) => Some(entry),
                Err(e) => {
                    log::warn!("Failed to encode mask data: {e}");
                    None
                }
            }
        } else {
            None
        }
    }

    fn get_created_files(&self) -> Vec<std::path::PathBuf> {
        self.created_files.clone()
    }
}

/// Cutout processor implementing the generic ModelProcessor trait
pub struct CutoutProcessor;

impl ModelProcessor for CutoutProcessor {
    type Config = CutoutConfig;
    type Result = CutoutResult;

    fn get_model_source<'a>(config: &Self::Config) -> Result<ModelSource<'a>> {
        // Create CLI model info from config
        let cli_model_info = crate::model_access::CliModelInfo {
            model_path: config.model_path.clone(),
            model_url: config.model_url.clone(),
            model_checksum: config.model_checksum.clone(),
        };

        // Use CLI-aware model access
        CutAccess::get_model_source_with_cli(&cli_model_info)
    }

    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
    ) -> Result<Self::Result> {
        let start_time = Instant::now();
        let mut io_timing = IoTiming::new();

        debug!("🖼️  Processing: {}", image_path.display());

        // Load and preprocess the image with timing
        let img = io_timing.time_image_read(image_path)?;
        let original_size = img.dimensions();

        let input_array = preprocess_image_for_isnet_v2(&img)?;

        // Prepare input for the model
        let input_name = session.inputs[0].name.clone();
        let output_name = session.outputs[0].name.clone();
        let input_value = Value::from_array(input_array)
            .map_err(|e| anyhow::anyhow!("Failed to create input value: {}", e))?;

        // Run inference
        let outputs = session
            .run(ort::inputs![input_name.as_str() => &input_value])
            .map_err(|e| anyhow::anyhow!("Failed to run inference: {}", e))?;

        // Extract the output tensor using ORT v2 API
        let output_view = outputs[output_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract output array: {}", e))?;

        // Extract the mask from the output (shape should be [1, 1, 1024, 1024])
        let mask_2d = output_view.slice(ndarray::s![0, 0, .., ..]);

        // Post-process the mask
        let mask = postprocess_mask(&mask_2d, original_size, config.post_process_mask)?;

        // Extract binary mask data for metadata (threshold at 128)
        let raw_mask_data = extract_binary_mask_data(&mask);

        // Generate output paths using OutputManager
        let output_manager = OutputManager::new(config, image_path);
        let output_path = output_manager.generate_main_output_path("cutout", "png")?;
        let mask_path = if config.save_mask {
            Some(output_manager.generate_auxiliary_output("mask", "png")?)
        } else {
            None
        };

        // Track created files
        let mut created_files = Vec::new();

        // Create the cutout
        let cutout_result = if config.alpha_matting {
            apply_alpha_matting(
                &img,
                &mask,
                config.alpha_matting_foreground_threshold,
                config.alpha_matting_background_threshold,
                config.alpha_matting_erode_size,
            )?
        } else if let Some(bg_color) = config.background_color {
            create_cutout_with_background(&img, &mask, bg_color)?
        } else {
            create_cutout(&img, &mask)?
        };

        // Save the cutout (always PNG for transparency) with timing
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)?;
        }
        io_timing.time_save_operation(|| Ok(cutout_result.save(&output_path)?))?;

        // Track the main output as produced
        output_manager.track_output(output_path.clone());
        created_files.push(output_path.clone());

        debug!(
            "{} Cutout saved to: {}",
            symbols::completed_successfully(),
            output_path.display()
        );
        // Save mask if requested with timing
        if let Some(mask_path_val) = &mask_path {
            if let Some(parent) = Path::new(mask_path_val).parent() {
                fs::create_dir_all(parent)?;
            }
            io_timing.time_save_operation(|| Ok(mask.save(mask_path_val)?))?;

            // Track the mask output as produced
            output_manager.track_output(mask_path_val.clone());
            created_files.push(mask_path_val.clone());

            debug!(
                "{} Mask saved to: {}",
                symbols::completed_successfully(),
                mask_path_val.display()
            );
        }

        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Create result with timing information
        let cutout_result = CutoutResult {
            output_path: output_path.to_string_lossy().to_string(),
            model_version: get_default_cutout_model_info().name,
            processing_time_ms: processing_time,
            mask_path: mask_path.map(|p| p.to_string_lossy().to_string()),
            io_timing,
            raw_mask_data: Some(raw_mask_data),
            created_files,
        };

        Ok(cutout_result)
    }

    fn serialize_config(config: &Self::Config) -> Result<toml::Value> {
        Ok(toml::Value::try_from(config)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_access_env_vars() {
        // Test environment variable names
        assert_eq!(CutAccess::get_env_var_name(), "BEAKER_CUTOUT_MODEL_PATH");
        assert_eq!(
            CutAccess::get_url_env_var_name(),
            Some("BEAKER_CUTOUT_MODEL_URL")
        );
        assert_eq!(
            CutAccess::get_checksum_env_var_name(),
            Some("BEAKER_CUTOUT_MODEL_CHECKSUM")
        );
    }

    #[test]
    fn test_cut_access_no_embedded_bytes() {
        let bytes = CutAccess::get_embedded_bytes();
        assert!(
            bytes.is_none(),
            "Cutout model should not have embedded bytes"
        );
    }

    #[test]
    fn test_cut_access_has_default_info() {
        let model_info = CutAccess::get_default_model_info();
        assert!(
            model_info.is_some(),
            "Cutout model should have default model info"
        );

        let info = model_info.unwrap();
        assert_eq!(info.name, "isnet-general-use-v1");
        assert!(info.url.contains("isnet-general-use.onnx"));
        assert!(!info.md5_checksum.is_empty());
        assert_eq!(info.filename, "isnet-general-use.onnx");
    }

    // Environment variable tests are now in integration tests to avoid race conditions

    #[test]
    fn test_get_default_cutout_model_info() {
        let model_info = get_default_cutout_model_info();
        assert_eq!(model_info.name, "isnet-general-use-v1");
        assert!(model_info.url.contains("isnet-general-use.onnx"));
        assert!(!model_info.md5_checksum.is_empty());
        assert_eq!(model_info.filename, "isnet-general-use.onnx");
    }

    #[test]
    fn test_runtime_model_info_with_cutout_overrides() {
        use crate::model_access::RuntimeModelInfo;

        // Test RuntimeModelInfo creation without env var modification
        let default_info = get_default_cutout_model_info();

        // Test without any env vars (should use default info)
        let runtime_info = RuntimeModelInfo::from_model_info_with_overrides(
            &default_info,
            Some("NONEXISTENT_TEST_URL"),
            Some("NONEXISTENT_TEST_CHECKSUM"),
        );

        assert_eq!(runtime_info.name, default_info.name);
        assert_eq!(runtime_info.url, default_info.url);
        assert_eq!(runtime_info.md5_checksum, default_info.md5_checksum);
        assert_eq!(runtime_info.filename, default_info.filename);
    }
}
