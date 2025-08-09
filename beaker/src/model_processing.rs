//! Model processing framework providing unified interface for all models.
//!
//! This module defines the core `ModelProcessor` trait that all models must implement,
//! along with generic processing functions that handle the common patterns across
//! different model types.

use crate::progress::{add_progress_bar, remove_progress_bar};
use anyhow::Result;
use colored::Colorize;
use log::warn;
use ort::session::Session;
use std::path::Path;
use std::time::Duration;

use crate::config::BaseModelConfig;
use crate::image_input::{collect_images_from_sources, ImageInputConfig};
use crate::onnx_session::{create_onnx_session, ModelSource, SessionConfig};

use crate::color_utils::maybe_dim_stderr;
use crate::shared_metadata::{InputProcessing, SystemInfo};

/// Configuration trait for models that can be processed generically
pub trait ModelConfig: std::any::Any {
    fn base(&self) -> &BaseModelConfig;

    /// Get the tool name for this config (e.g., "detect", "cutout")
    fn tool_name(&self) -> &'static str;
}

/// Result trait for model outputs that can be handled generically
pub trait ModelResult {
    /// Get the processing time in milliseconds
    fn processing_time_ms(&self) -> f64;

    /// Get the tool name for metadata sections (e.g., "head", "cutout")
    fn tool_name(&self) -> &'static str;

    /// Get the serializable core results for the main tool section
    fn core_results(&self) -> Result<toml::Value>;

    /// Get a summary of all output files created
    fn output_summary(&self) -> String;

    /// Get file I/O timing information
    fn get_io_timing(&self) -> crate::shared_metadata::IoTiming;

    /// Get list of files actually created (for depfile generation)
    fn get_created_files(&self) -> Vec<std::path::PathBuf>;

    /// Get mask entry for cutout results (only applicable for cutout tools)
    fn get_mask_entry(&self) -> Option<crate::mask_encoding::MaskEntry> {
        None
    }
}

/// Core trait that all models must implement
pub trait ModelProcessor {
    /// Configuration type for this model
    type Config: ModelConfig;

    /// Result type returned by this model
    type Result: ModelResult;

    /// Get the model source for loading the ONNX model
    fn get_model_source<'a>(config: &Self::Config) -> Result<ModelSource<'a>>;

    /// Process a single image through the complete pipeline
    fn process_single_image(
        session: &mut Session,
        image_path: &Path,
        config: &Self::Config,
    ) -> Result<Self::Result>;

    /// Get serializable configuration for metadata
    fn serialize_config(config: &Self::Config) -> Result<toml::Value>;
}

/// Generic batch processing function that works with any model
pub fn run_model_processing<P: ModelProcessor>(config: P::Config) -> Result<usize> {
    use crate::onnx_session::determine_optimal_device;
    use chrono::Utc;
    use std::time::Instant;

    let start_timestamp = Utc::now();
    let total_processing_start = Instant::now();

    // Collect command line for metadata
    let command_line: Vec<String> = std::env::args().collect();

    // Create image input configuration from model config
    let image_config = ImageInputConfig::from_strict_flag(config.base().strict);

    // Collect images from input sources
    let image_files = collect_images_from_sources(&config.base().sources, &image_config)?;

    if image_files.is_empty() {
        log::warn!("No valid images found to process");
        return Ok(0);
    }

    if image_files.len() > 1 {
        log::info!(
            "{} Found {} images to process",
            crate::color_utils::symbols::resources_found(),
            image_files.len()
        );
    }

    // Collect device information for metadata
    let device_selection = determine_optimal_device(&config.base().device);
    let device_selected = device_selection.device.clone();
    let device_selection_reason = device_selection.reason.clone();

    // Create session with timing
    let session_start = Instant::now();
    // It sometimes takes a while to load the model, so we'll use a spinner when on TTY
    // it will show "model loading" while the model is being loaded, then switch to "model loaded" when done
    let spinner = indicatif::ProgressBar::new_spinner();
    add_progress_bar(spinner.clone());
    spinner.set_message(" Loading model...");
    spinner.enable_steady_tick(Duration::from_millis(100));

    let model_source = P::get_model_source(&config)?;

    let session_config = SessionConfig {
        device: &device_selected,
    };
    let (mut session, model_info) = create_onnx_session(model_source, &session_config)?;

    spinner.finish_and_clear();
    remove_progress_bar(&spinner);

    let model_load_time_ms = session_start.elapsed().as_secs_f64() * 1000.0;

    let megabytes_str = format!("{} MB", model_info.model_size_bytes / 1_048_576);

    let pretty_device_name = if device_selected == "coreml" {
        "CoreML".to_string()
    } else {
        "CPU".to_string()
    };

    let timing_str: String = maybe_dim_stderr(&format!("in {model_load_time_ms:.0}ms"));
    log::info!(
        "{} Model loaded ({}, {}) {}",
        crate::color_utils::symbols::model_loaded(),
        megabytes_str,
        pretty_device_name,
        timing_str
    );

    let system = SystemInfo {
        device_requested: Some(config.base().device.clone()),
        device_selected: Some(device_selected.to_string()),
        device_selection_reason: Some(device_selection_reason.to_string()),
        execution_providers: model_info.execution_providers,
        model_source: Some(model_info.model_source),
        model_path: model_info.model_path.clone(),
        model_size_bytes: Some(model_info.model_size_bytes.try_into().unwrap()),
        model_load_time_ms: Some(model_load_time_ms),
        model_checksum: Some(model_info.model_checksum),
    };

    // Generate stamps for Make-compatible incremental builds if depfile is requested
    let stamp_info = if config.base().depfile.is_some() {
        let model_path = model_info.model_path.as_ref().map(Path::new);
        Some(generate_stamps_for_tool::<P>(&config, model_path)?)
    } else {
        None
    };

    // Process each image and collect results
    let mut successful_count = 0;
    let mut failed_count = 0;

    // Create progress bar for batch processing if appropriate
    let progress_bar: Option<indicatif::ProgressBar> =
        crate::color_utils::progress::create_batch_progress_bar(image_files.len());

    // Create vector to contain failed image paths
    let mut failed_images = Vec::new();
    for (index, (image_path, (source_type, source_string))) in image_files.iter().enumerate() {
        // Create input processing info
        let input = InputProcessing {
            image_path: image_path.to_string_lossy().to_string(),
            source: source_string.to_string(),
            source_type: source_type.to_string(),
            strict_mode: config.base().strict,
        };

        if let Some(ref pb) = progress_bar {
            // we will style the filename with bold:
            let filename = crate::color_utils::maybe_color_stderr(
                &image_path.file_name().unwrap_or_default().to_string_lossy(),
                |s| s.bold(),
            );
            pb.set_prefix(format!(
                "[{}/{}] Processing {}",
                index + 1,
                image_files.len(),
                filename,
            ));
            // we are using msg as ETA
            if index > 0 {
                // Calculate ETA based on elapsed time and number of processed images
                // This is a simple linear estimate, not perfect but works for most cases
                let elapsed = pb.elapsed().as_secs_f64();
                let total = image_files.len() as f64;
                let processed = index as f64; // no +1 since we haven't processed this one yet
                let eta = (elapsed / processed) * (total - processed);
                pb.set_message(format!("ETA: {eta:.1}s"));
            }
        }
        match P::process_single_image(&mut session, image_path, &config) {
            Ok(result) => {
                successful_count += 1;

                if !config.base().skip_metadata {
                    save_enhanced_metadata_for_file::<P>(
                        &result,
                        &config,
                        image_path,
                        &command_line,
                        system.clone(),
                        input.clone(),
                        start_timestamp,
                        stamp_info.as_ref(),
                    )?;
                }
                if progress_bar.is_none() {
                    let val = result.processing_time_ms();
                    let timing_str: String = maybe_dim_stderr(&format!("in {val:.0} ms"));
                    // Log comprehensive processing result for single files or non-interactive
                    log::info!(
                        "{} Processed {} ({}/{}) {} {}",
                        crate::color_utils::symbols::completed_successfully(),
                        image_path.display(),
                        index + 1,
                        image_files.len(),
                        timing_str,
                        result.output_summary()
                    );
                }
            }
            Err(e) => {
                failed_count += 1;

                failed_images.push(image_path.to_str().unwrap_or_default().to_string());

                let colored_error = crate::color_utils::colors::error_level(&e.to_string());
                warn!(
                    "{} Failed to process {}:\n            {}",
                    crate::color_utils::symbols::warning(),
                    image_path.display(),
                    colored_error
                );
            }
        }
        if let Some(ref pb) = progress_bar {
            pb.inc(1);
        }
    }

    // Finish progress bar if it exists
    if let Some(ref pb) = progress_bar {
        pb.finish_and_clear();
        remove_progress_bar(pb);
    }
    let total_processing_time = total_processing_start.elapsed();

    // The case n=1 doesn't use a progress bar and already got a direct log message
    if image_files.len() > 1 {
        if failed_count > 0 {
            let timing_str = maybe_dim_stderr(&format!(
                "({:.1}s, {:.0} ms/success)",
                total_processing_time.as_millis() as f64 / 1000.0,
                total_processing_time.as_millis() as f64 / successful_count as f64
            ));
            log::info!(
                "{} Processed {} images with {} successes and {} failures {}",
                crate::color_utils::symbols::completed_partially_successfully(),
                successful_count + failed_count,
                successful_count,
                failed_count,
                timing_str
            );
        } else {
            let timing_str = maybe_dim_stderr(&format!(
                "({:.1}s, {:.0} ms/image)",
                total_processing_time.as_millis() as f64 / 1000.0,
                total_processing_time.as_millis() as f64 / successful_count as f64
            ));
            log::info!(
                "{} Processed {} images successfully {}",
                crate::color_utils::symbols::completed_successfully(),
                successful_count,
                timing_str
            );
        }
    }

    // If strict mode is enabled, fail if any images failed
    if config.base().strict && failed_count > 0 {
        return Err(anyhow::anyhow!(
            "{} image(s) failed to process (without `--permissive` flag)",
            failed_count
        ));
    }
    Ok(successful_count)
}

/// Save enhanced metadata for a single processed file
#[allow(clippy::too_many_arguments)]
fn save_enhanced_metadata_for_file<P: ModelProcessor>(
    result: &P::Result,
    config: &P::Config,
    image_path: &std::path::Path,
    command_line: &[String],
    system: SystemInfo,
    input: InputProcessing,
    start_timestamp: chrono::DateTime<chrono::Utc>,
    stamp_info: Option<&crate::stamp_manager::StampInfo>,
) -> Result<()> {
    use crate::output_manager::OutputManager;
    use crate::shared_metadata::{CutoutSections, DetectSections, ExecutionContext};

    let output_manager = OutputManager::new(config, image_path);

    // Create execution context
    let execution = ExecutionContext {
        timestamp: Some(start_timestamp),
        beaker_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        command_line: Some(command_line.to_vec()),
        exit_code: Some(0),
        model_processing_time_ms: Some(result.processing_time_ms()),
        file_io: Some(result.get_io_timing()),
        beaker_env_vars: crate::shared_metadata::collect_beaker_env_vars(),
    };

    // Get core results and config
    let core_results = result.core_results()?;
    let config_value = P::serialize_config(config)?;

    // Create the appropriate sections based on tool type
    match result.tool_name() {
        "detect" => {
            let detect_sections = DetectSections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
            };
            output_manager.save_complete_metadata(Some(detect_sections), None)?;
        }
        "cutout" => {
            let cutout_sections = CutoutSections {
                core: Some(core_results),
                config: Some(config_value),
                execution: Some(execution),
                system: Some(system),
                input: Some(input),
                mask: result.get_mask_entry(),
            };
            output_manager.save_complete_metadata(None, Some(cutout_sections))?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown tool name: {}", result.tool_name()));
        }
    }

    // Generate depfile if requested and stamps are available
    if let (Some(depfile_path), Some(stamp_info)) = (&config.base().depfile, stamp_info) {
        generate_depfile_for_image::<P>(
            result,
            config,
            image_path,
            depfile_path,
            stamp_info,
            &output_manager,
        )?;
    }

    Ok(())
}

/// Generate stamps for the appropriate tool type
fn generate_stamps_for_tool<P: ModelProcessor>(
    config: &P::Config,
    model_path: Option<&Path>,
) -> Result<crate::stamp_manager::StampInfo> {
    match config.tool_name() {
        "detect" => {
            // Cast to DetectionConfig
            let detection_config = config as &dyn std::any::Any;
            let detection_config = detection_config
                .downcast_ref::<crate::config::DetectionConfig>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast to DetectionConfig"))?;
            crate::stamp_manager::generate_detection_stamps(detection_config, model_path)
        }
        "cutout" => {
            // Cast to CutoutConfig
            let cutout_config = config as &dyn std::any::Any;
            let cutout_config = cutout_config
                .downcast_ref::<crate::config::CutoutConfig>()
                .ok_or_else(|| anyhow::anyhow!("Failed to downcast to CutoutConfig"))?;
            crate::stamp_manager::generate_cutout_stamps(cutout_config, model_path)
        }
        tool_name => Err(anyhow::anyhow!("Unknown tool name: {}", tool_name)),
    }
}

/// Generate a depfile for a single processed image
fn generate_depfile_for_image<P: ModelProcessor>(
    result: &P::Result,
    _config: &P::Config,
    image_path: &Path,
    depfile_path: &str,
    stamp_info: &crate::stamp_manager::StampInfo,
    output_manager: &crate::output_manager::OutputManager,
) -> Result<()> {
    use crate::depfile_generator::generate_depfile;

    // Input files are just the source image
    let inputs = vec![image_path.to_path_buf()];

    // Combine result's created files with metadata files tracked by OutputManager
    let mut outputs = result.get_created_files();
    let metadata_files = output_manager.get_produced_outputs();
    outputs.extend(metadata_files);

    // Generate the depfile
    generate_depfile(Path::new(depfile_path), &outputs, &inputs, stamp_info)?;

    Ok(())
}
