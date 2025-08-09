//! Output path management providing unified logic for all models.
//!
//! This module handles the complexities of output path generation across different
//! models, ensuring consistent behavior for:
//! - Single vs multiple outputs
//! - Output directory vs same directory as input
//! - Suffix handling for auxiliary files
//! - Numbered outputs with appropriate zero-padding
//! - Metadata path utilities

use anyhow::Result;
use log::debug;
use std::path::{Path, PathBuf};

use crate::model_processing::ModelConfig;
use crate::shared_metadata::{
    get_metadata_path, load_or_create_metadata, save_metadata, CutoutSections, DetectSections,
};

/// Unified output path management for all models
pub struct OutputManager<'a> {
    config: &'a dyn ModelConfig,
    input_path: &'a Path,
    produced_outputs: std::cell::RefCell<Vec<PathBuf>>,
}

impl<'a> OutputManager<'a> {
    /// Create a new OutputManager for the given config and input path
    pub fn new(config: &'a dyn ModelConfig, input_path: &'a Path) -> Self {
        Self {
            config,
            input_path,
            produced_outputs: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// Track that an output file was produced
    pub fn track_output(&self, path: PathBuf) {
        self.produced_outputs.borrow_mut().push(path);
    }

    /// Get all outputs that have been tracked as produced
    pub fn get_produced_outputs(&self) -> Vec<PathBuf> {
        self.produced_outputs.borrow().clone()
    }

    /// Get the input file stem (filename without extension)
    fn input_stem(&self) -> &str {
        self.input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
    }

    /// Generate primary output path with suffix (always includes suffix)
    ///
    /// This ensures consistent naming regardless of output directory usage
    pub fn generate_main_output_path(
        &self,
        default_suffix: &str,
        extension: &str,
    ) -> Result<PathBuf> {
        let input_stem = self.input_stem();

        // Always add suffix for consistency
        let output_filename = format!("{input_stem}_{default_suffix}.{extension}");

        let output_path = if let Some(output_dir) = &self.config.base().output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(&output_filename)
        } else {
            self.input_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&output_filename)
        };

        Ok(output_path)
    }

    /// Generate numbered output path for multiple similar outputs
    ///
    /// Examples:
    /// - Single item: "image_crop.jpg" (always with suffix)
    /// - Multiple items < 10: "image_crop-1.jpg", "image_crop-2.jpg"
    /// - Multiple items >= 10: "image_crop-01.jpg", "image_crop-02.jpg"
    pub fn generate_numbered_output(
        &self,
        base_suffix: &str,
        index: usize,
        total: usize,
        extension: &str,
    ) -> Result<PathBuf> {
        let input_stem = self.input_stem();

        let output_filename = if total == 1 {
            // Single output - always use suffix for consistency
            format!("{input_stem}_{base_suffix}.{extension}")
        } else {
            // Multiple outputs - always numbered with suffix
            let number_format = if total >= 10 {
                format!("{index:02}") // Zero-padded for 10+
            } else {
                format!("{index}") // No padding for < 10
            };

            format!("{input_stem}_{base_suffix}-{number_format}.{extension}")
        };

        let output_path = if let Some(output_dir) = &self.config.base().output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(&output_filename)
        } else {
            self.input_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&output_filename)
        };

        Ok(output_path)
    }

    /// Generate auxiliary output path (always includes suffix)
    pub fn generate_auxiliary_output(&self, suffix: &str, extension: &str) -> Result<PathBuf> {
        let input_stem = self.input_stem();
        let output_filename = format!("{input_stem}_{suffix}.{extension}");

        let output_path = if let Some(output_dir) = &self.config.base().output_dir {
            let output_dir = Path::new(output_dir);
            std::fs::create_dir_all(output_dir)?;
            output_dir.join(&output_filename)
        } else {
            self.input_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&output_filename)
        };

        Ok(output_path)
    }

    /// Generate main output path and track it as produced
    pub fn generate_and_track_main_output(
        &self,
        default_suffix: &str,
        extension: &str,
    ) -> Result<PathBuf> {
        let path = self.generate_main_output_path(default_suffix, extension)?;
        self.track_output(path.clone());
        Ok(path)
    }

    /// Generate numbered output path and track it as produced
    pub fn generate_and_track_numbered_output(
        &self,
        base_suffix: &str,
        index: usize,
        total: usize,
        extension: &str,
    ) -> Result<PathBuf> {
        let path = self.generate_numbered_output(base_suffix, index, total, extension)?;
        self.track_output(path.clone());
        Ok(path)
    }

    /// Generate auxiliary output path and track it as produced
    pub fn generate_and_track_auxiliary_output(
        &self,
        suffix: &str,
        extension: &str,
    ) -> Result<PathBuf> {
        let path = self.generate_auxiliary_output(suffix, extension)?;
        self.track_output(path.clone());
        Ok(path)
    }

    /// Make a file path relative to the metadata file location
    pub fn make_relative_to_metadata(&self, path: &Path) -> Result<String> {
        if self.config.base().skip_metadata {
            return Ok(path.to_string_lossy().to_string());
        }

        let metadata_path =
            get_metadata_path(self.input_path, self.config.base().output_dir.as_deref())?;

        make_path_relative_to_toml(path, &metadata_path)
    }

    /// Save complete metadata sections (core + enhanced sections)
    pub fn save_complete_metadata(
        &self,
        detect_sections: Option<DetectSections>,
        cutout_sections: Option<CutoutSections>,
    ) -> Result<()> {
        if self.config.base().skip_metadata {
            return Ok(());
        }

        let metadata_path =
            get_metadata_path(self.input_path, self.config.base().output_dir.as_deref())?;

        let mut metadata = load_or_create_metadata(&metadata_path)?;

        // Update the sections that were provided
        if let Some(detect) = detect_sections {
            metadata.detect = Some(detect);
        }
        if let Some(cutout) = cutout_sections {
            metadata.cutout = Some(cutout);
        }

        save_metadata(&metadata, &metadata_path)?;

        // Track the metadata file as an output
        self.track_output(metadata_path.clone());

        debug!("📋 Saved complete metadata to: {}", metadata_path.display());

        Ok(())
    }
}

/// Make a file path relative to a TOML file (used for metadata)
pub fn make_path_relative_to_toml(file_path: &Path, toml_path: &Path) -> Result<String> {
    if let Some(toml_dir) = toml_path.parent() {
        if let Ok(rel_path) = file_path.strip_prefix(toml_dir) {
            // Convert to forward slashes for cross-platform compatibility
            Ok(rel_path.to_string_lossy().replace('\\', "/"))
        } else {
            // If we can't make it relative, use absolute path
            Ok(file_path.to_string_lossy().to_string())
        }
    } else {
        Ok(file_path.to_string_lossy().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BaseModelConfig, DetectionConfig};
    use std::collections::HashSet;
    use tempfile::TempDir;

    fn create_test_config(output_dir: Option<String>) -> DetectionConfig {
        DetectionConfig {
            base: BaseModelConfig {
                sources: vec!["test.jpg".to_string()],
                device: "cpu".to_string(),
                output_dir,
                depfile: None,
                skip_metadata: false,
                strict: true,
            },
            confidence: 0.25,
            iou_threshold: 0.45,
            crop_classes: HashSet::new(), // Empty for this test
            bounding_box: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
        }
    }

    #[test]
    fn test_main_output_path_same_directory() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager.generate_main_output_path("cutout", "png").unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_cutout.png"));
    }

    #[test]
    fn test_main_output_path_with_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager.generate_main_output_path("cutout", "png").unwrap();

        // Should now always include suffix, even with output_dir
        assert_eq!(output_path, output_dir.join("test_cutout.png"));
    }

    #[test]
    fn test_auxiliary_output_always_has_suffix() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager.generate_auxiliary_output("mask", "png").unwrap();

        assert_eq!(output_path, output_dir.join("test_mask.png"));
    }

    #[test]
    fn test_numbered_output_single_item() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 1, 1, "jpg")
            .unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_crop.jpg"));
    }

    #[test]
    fn test_numbered_output_single_item_with_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 1, 1, "jpg")
            .unwrap();

        // Should now always include suffix, even with output_dir and single item
        assert_eq!(output_path, output_dir.join("test_crop.jpg"));
    }

    #[test]
    fn test_numbered_output_multiple_items_less_than_10() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 2, 5, "jpg")
            .unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_crop-2.jpg"));
    }

    #[test]
    fn test_numbered_output_multiple_items_10_or_more() {
        let temp_dir = TempDir::new().unwrap();
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(None);

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 5, 15, "jpg")
            .unwrap();

        assert_eq!(output_path, temp_dir.path().join("test_crop-05.jpg"));
    }

    #[test]
    fn test_numbered_output_with_output_dir() {
        let temp_dir = TempDir::new().unwrap();
        let output_dir = temp_dir.path().join("output");
        let input_path = temp_dir.path().join("test.jpg");
        let config = create_test_config(Some(output_dir.to_string_lossy().to_string()));

        let manager = OutputManager::new(&config, &input_path);
        let output_path = manager
            .generate_numbered_output("crop", 3, 12, "jpg")
            .unwrap();

        // Should now always include suffix, even with output_dir
        assert_eq!(output_path, output_dir.join("test_crop-03.jpg"));
    }

    #[test]
    fn test_make_path_relative_to_toml() {
        let temp_dir = TempDir::new().unwrap();
        let toml_path = temp_dir.path().join("test.beaker.toml");
        let file_path = temp_dir.path().join("test_crop.jpg");

        let relative = make_path_relative_to_toml(&file_path, &toml_path).unwrap();
        assert_eq!(relative, "test_crop.jpg");
    }

    #[test]
    fn test_make_path_relative_with_subdirectory() {
        let temp_dir = TempDir::new().unwrap();
        let subdir = temp_dir.path().join("subdir");
        let toml_path = temp_dir.path().join("test.beaker.toml");
        let file_path = subdir.join("test_crop.jpg");

        let relative = make_path_relative_to_toml(&file_path, &toml_path).unwrap();
        assert_eq!(relative, "subdir/test_crop.jpg");
    }
}
