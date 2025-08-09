//! Stamp file management for Make-compatible incremental builds
//!
//! This module handles the creation and management of stamp files that enable
//! Make to perform accurate incremental builds. Each stamp file contains a
//! deterministic hash based only on inputs that affect the byte-level output
//! of that stage.

use anyhow::{anyhow, Result};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cache_common::{calculate_md5, get_cache_base_dir};
use crate::config::{CutoutConfig, DetectionConfig};

/// Generate a deterministic configuration hash for detection
///
/// **IMPORTANT**: This function must be kept in sync with DetectionConfig.
/// When adding new fields to DetectionConfig that affect byte-level output:
/// 1. Add the field to the hash computation below
/// 2. Update tests in stamp_manager.rs
/// 3. Consider if the field should be included in serialized config
///
/// Only include parameters that affect the actual image bytes produced,
/// not metadata-only or performance-related settings.
pub fn generate_detection_config_hash(config: &DetectionConfig) -> String {
    let mut hasher = Sha256::new();

    // Only include parameters that affect byte-level output
    hasher.update(format!("confidence:{}", config.confidence));
    hasher.update(format!("iou_threshold:{}", config.iou_threshold));

    // Sort crop classes for deterministic hash
    let mut crop_classes: Vec<_> = config
        .crop_classes
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();
    crop_classes.sort();
    hasher.update(format!("crop_classes:{crop_classes:?}"));

    hasher.update(format!("bounding_box:{}", config.bounding_box));

    // Include custom model parameters if specified
    if let Some(ref model_path) = config.model_path {
        hasher.update(format!("model_path:{model_path}"));
    }
    if let Some(ref model_url) = config.model_url {
        hasher.update(format!("model_url:{model_url}"));
    }
    if let Some(ref model_checksum) = config.model_checksum {
        hasher.update(format!("model_checksum:{model_checksum}"));
    }

    // Include output directory if it affects output file names/paths
    if let Some(ref output_dir) = config.base.output_dir {
        hasher.update(format!("output_dir:{output_dir}"));
    }

    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Generate a deterministic configuration hash for cutout processing
///
/// **IMPORTANT**: This function must be kept in sync with CutoutConfig.
/// When adding new fields to CutoutConfig that affect byte-level output:
/// 1. Add the field to the hash computation below
/// 2. Update tests in stamp_manager.rs
/// 3. Consider if the field should be included in serialized config
///
/// Only include parameters that affect the actual image bytes produced,
/// not metadata-only or performance-related settings.
pub fn generate_cutout_config_hash(config: &CutoutConfig) -> String {
    let mut hasher = Sha256::new();

    // Only include parameters that affect byte-level output
    hasher.update(format!("post_process_mask:{}", config.post_process_mask));
    hasher.update(format!("alpha_matting:{}", config.alpha_matting));
    hasher.update(format!(
        "alpha_matting_foreground_threshold:{}",
        config.alpha_matting_foreground_threshold
    ));
    hasher.update(format!(
        "alpha_matting_background_threshold:{}",
        config.alpha_matting_background_threshold
    ));
    hasher.update(format!(
        "alpha_matting_erode_size:{}",
        config.alpha_matting_erode_size
    ));
    hasher.update(format!("save_mask:{}", config.save_mask));

    if let Some(bg_color) = config.background_color {
        hasher.update(format!("background_color:{bg_color:?}"));
    }

    // Include custom model parameters if specified
    if let Some(ref model_path) = config.model_path {
        hasher.update(format!("model_path:{model_path}"));
    }
    if let Some(ref model_url) = config.model_url {
        hasher.update(format!("model_url:{model_url}"));
    }
    if let Some(ref model_checksum) = config.model_checksum {
        hasher.update(format!("model_checksum:{model_checksum}"));
    }

    // Include output directory if it affects output file names/paths
    if let Some(ref output_dir) = config.base.output_dir {
        hasher.update(format!("output_dir:{output_dir}"));
    }

    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Generate tool version hash
pub fn generate_tool_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("beaker:{}", env!("CARGO_PKG_VERSION")));
    format!("{:x}", hasher.finalize())[..16].to_string()
}

/// Generate model hash from file path
pub fn generate_model_hash_from_path(model_path: &Path) -> Result<String> {
    if model_path.exists() {
        calculate_md5(model_path).map(|hash| hash[..16].to_string())
    } else {
        Err(anyhow!("Model file not found: {}", model_path.display()))
    }
}

/// Get the beaker stamps cache directory
pub fn get_stamps_cache_dir() -> Result<PathBuf> {
    let cache_dir = get_cache_base_dir()?;
    Ok(cache_dir.join("beaker").join("stamps"))
}

/// Create stamp file if content has changed
pub fn create_or_update_stamp(stamp_type: &str, hash: &str, content: &str) -> Result<PathBuf> {
    let stamps_dir = get_stamps_cache_dir()?;
    fs::create_dir_all(&stamps_dir)?;

    let stamp_filename = format!("{stamp_type}-{hash}.stamp");
    let stamp_path = stamps_dir.join(stamp_filename);

    // Check if stamp already exists with same content
    if stamp_path.exists() {
        if let Ok(existing_content) = fs::read_to_string(&stamp_path) {
            if existing_content == content {
                // Content is identical, preserve mtime
                return Ok(stamp_path);
            }
        }
    }

    // Write new content atomically
    let temp_path = stamp_path.with_extension("tmp");
    fs::write(&temp_path, content)?;
    fs::rename(temp_path, &stamp_path)?;

    Ok(stamp_path)
}

/// Generate all stamps for a detection run
pub fn generate_detection_stamps(
    config: &DetectionConfig,
    model_path: Option<&Path>,
) -> Result<StampInfo> {
    let config_hash = generate_detection_config_hash(config);
    let tool_hash = generate_tool_hash();

    // Generate config stamp
    let config_content = format!("detection-config:{config_hash}");
    let config_stamp = create_or_update_stamp("cfg-detect", &config_hash, &config_content)?;

    // Generate tool stamp
    let tool_content = format!("tool:{tool_hash}");
    let tool_stamp = create_or_update_stamp("tool", &tool_hash, &tool_content)?;

    // Generate model stamp if model file exists
    let model_stamp = if let Some(model_path) = model_path {
        let model_hash = generate_model_hash_from_path(model_path)?;
        let model_content = format!("model-detect:{model_hash}");
        Some(create_or_update_stamp(
            "model-detect",
            &model_hash,
            &model_content,
        )?)
    } else {
        None
    };

    Ok(StampInfo {
        config_stamp,
        tool_stamp,
        model_stamp,
    })
}

/// Generate all stamps for a cutout run
pub fn generate_cutout_stamps(
    config: &CutoutConfig,
    model_path: Option<&Path>,
) -> Result<StampInfo> {
    let config_hash = generate_cutout_config_hash(config);
    let tool_hash = generate_tool_hash();

    // Generate config stamp
    let config_content = format!("cutout-config:{config_hash}");
    let config_stamp = create_or_update_stamp("cfg-cutout", &config_hash, &config_content)?;

    // Generate tool stamp
    let tool_content = format!("tool:{tool_hash}");
    let tool_stamp = create_or_update_stamp("tool", &tool_hash, &tool_content)?;

    // Generate model stamp if model file exists
    let model_stamp = if let Some(model_path) = model_path {
        let model_hash = generate_model_hash_from_path(model_path)?;
        let model_content = format!("model-cutout:{model_hash}");
        Some(create_or_update_stamp(
            "model-cutout",
            &model_hash,
            &model_content,
        )?)
    } else {
        None
    };

    Ok(StampInfo {
        config_stamp,
        tool_stamp,
        model_stamp,
    })
}

/// Information about generated stamp files
#[derive(Debug)]
pub struct StampInfo {
    pub config_stamp: PathBuf,
    pub tool_stamp: PathBuf,
    pub model_stamp: Option<PathBuf>,
}

impl StampInfo {
    /// Get all stamp paths as a vector
    pub fn all_stamps(&self) -> Vec<&Path> {
        let mut stamps = vec![self.config_stamp.as_path(), self.tool_stamp.as_path()];
        if let Some(ref model_stamp) = self.model_stamp {
            stamps.push(model_stamp.as_path());
        }
        stamps
    }
}

#[cfg(test)]
mod tests {
    //! Tests for stamp generation and config hashing
    //!
    //! **MAINTENANCE REMINDER**: When adding new fields to DetectionConfig or CutoutConfig
    //! that affect output file bytes, you MUST:
    //! 1. Update the corresponding hash function (generate_*_config_hash)
    //! 2. Add test cases here to verify the new field affects the hash
    //! 3. Verify that changes to the field cause different stamp files to be generated
    //!
    //! This helps prevent dependency tracking issues where changes don't trigger rebuilds.

    use super::*;
    use crate::config::{DetectCommand, GlobalArgs};
    use clap_verbosity_flag::Verbosity;

    fn create_test_detection_config() -> DetectionConfig {
        let global = GlobalArgs {
            device: "cpu".to_string(),
            output_dir: None,
            metadata: false,
            depfile: None,
            verbosity: Verbosity::new(0, 0),
            permissive: false,
            no_color: false,
        };

        let cmd = DetectCommand {
            sources: vec!["test.jpg".to_string()],
            confidence: 0.25,
            iou_threshold: 0.45,
            crop: Some("head".to_string()),
            bounding_box: false,
            model_path: None,
            model_url: None,
            model_checksum: None,
        };

        DetectionConfig::from_args(global, cmd).unwrap()
    }

    #[test]
    fn test_detection_config_hash_deterministic() {
        let config = create_test_detection_config();
        let hash1 = generate_detection_config_hash(&config);
        let hash2 = generate_detection_config_hash(&config);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 16); // Should be truncated to 16 chars
    }

    #[test]
    fn test_detection_config_hash_changes_with_params() {
        let mut config = create_test_detection_config();
        let hash1 = generate_detection_config_hash(&config);

        config.confidence = 0.5; // Change confidence
        let hash2 = generate_detection_config_hash(&config);

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_tool_hash_generation() {
        let hash1 = generate_tool_hash();
        let hash2 = generate_tool_hash();
        assert_eq!(hash1, hash2); // Should be deterministic
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn test_stamp_file_creation() {
        // Use a unique stamp type to avoid conflicts
        let stamp_type = format!("test-{}", std::process::id());
        let hash = "abcd1234";
        let content = "test content";

        let stamp_path = create_or_update_stamp(&stamp_type, hash, content).unwrap();

        assert!(stamp_path.exists());
        assert_eq!(fs::read_to_string(&stamp_path).unwrap(), content);

        // Cleanup
        let _ = fs::remove_file(&stamp_path);
    }

    #[test]
    fn test_stamp_file_preserves_mtime_when_unchanged() {
        let stamp_type = format!("test-preserve-{}", std::process::id());
        let hash = "efgh5678";
        let content = "preserve test content";

        // Create initial stamp
        let stamp_path = create_or_update_stamp(&stamp_type, hash, content).unwrap();
        let initial_metadata = fs::metadata(&stamp_path).unwrap();
        let initial_modified = initial_metadata.modified().unwrap();

        // Wait a bit to ensure time difference would be detectable
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Create stamp again with same content
        let stamp_path2 = create_or_update_stamp(&stamp_type, hash, content).unwrap();
        assert_eq!(stamp_path, stamp_path2);

        let final_metadata = fs::metadata(&stamp_path).unwrap();
        let final_modified = final_metadata.modified().unwrap();

        // Modification time should be preserved
        assert_eq!(initial_modified, final_modified);

        // Cleanup
        let _ = fs::remove_file(&stamp_path);
    }
}
