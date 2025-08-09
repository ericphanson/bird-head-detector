//! Make-compatible dependency file generation
//!
//! This module generates dependency files (.d files) that can be included
//! in Makefiles to enable accurate incremental builds. The depfiles list
//! all inputs that affect the byte-level output as prerequisites.
//!
//! ## New Architecture (Recommended)
//!
//! The new approach uses OutputManager to track actual outputs produced:
//! - `generate_depfile_from_output_manager()` uses OutputManager's tracked outputs
//! - This eliminates synchronization issues between depfile generation and actual outputs
//! - Single source of truth for what files are actually produced
//!
//! ## Legacy Functions (Deprecated)
//!
//! The functions `get_detection_output_files()` and `get_cutout_output_files()`
//! manually duplicate OutputManager logic and should not be used for new code.
//! They are kept for backwards compatibility and testing only.

use anyhow::Result;
use std::fs;
use std::path::{Path, PathBuf};

use crate::output_manager::OutputManager;
use crate::stamp_manager::StampInfo;

/// Generate a Make-compatible dependency file using OutputManager's tracked outputs
pub fn generate_depfile_from_output_manager(
    depfile_path: &Path,
    output_manager: &OutputManager,
    input_files: &[PathBuf],
    stamp_info: &StampInfo,
) -> Result<()> {
    let tracked_outputs = output_manager.get_produced_outputs();
    generate_depfile(depfile_path, &tracked_outputs, input_files, stamp_info)
}

/// Generate a Make-compatible dependency file
pub fn generate_depfile(
    depfile_path: &Path,
    targets: &[PathBuf],
    input_files: &[PathBuf],
    stamp_info: &StampInfo,
) -> Result<()> {
    // Collect all prerequisites: input files + stamp files
    let mut prerequisites = Vec::new();

    // Add input files
    for input_file in input_files {
        prerequisites.push(input_file.clone());
    }

    // Add stamp files
    for stamp_path in stamp_info.all_stamps() {
        prerequisites.push(stamp_path.to_path_buf());
    }

    // Generate depfile content
    let content = generate_depfile_content(targets, &prerequisites)?;

    // Write depfile atomically
    write_depfile_atomically(depfile_path, &content)?;

    Ok(())
}

/// Generate the content string for a depfile
fn generate_depfile_content(targets: &[PathBuf], prerequisites: &[PathBuf]) -> Result<String> {
    if targets.is_empty() {
        return Ok(String::new());
    }

    // Format targets
    let target_strs: Vec<String> = targets.iter().map(|p| escape_path_for_make(p)).collect();
    let targets_line = target_strs.join(" ");

    // Format prerequisites
    let prereq_strs: Vec<String> = prerequisites
        .iter()
        .map(|p| escape_path_for_make(p))
        .collect();
    let prereqs_line = prereq_strs.join(" ");

    // Generate Make rule
    let content = if prereqs_line.is_empty() {
        format!("{targets_line}:\n")
    } else {
        format!("{targets_line}: {prereqs_line}\n")
    };

    Ok(content)
}

/// Escape a path for use in Makefiles
///
/// Make requires spaces and some special characters to be escaped
fn escape_path_for_make(path: &Path) -> String {
    let path_str = path.to_string_lossy();

    // Escape spaces, dollar signs, and backslashes
    path_str
        .replace('\\', "\\\\") // Escape backslashes first
        .replace(' ', "\\ ") // Escape spaces
        .replace('$', "$$") // Escape dollar signs
}

/// Write depfile content atomically using temp file + rename
fn write_depfile_atomically(depfile_path: &Path, content: &String) -> Result<()> {
    // Create parent directory if needed
    if let Some(parent) = depfile_path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Write to temporary file first
    let temp_path = depfile_path.with_extension("tmp");
    fs::write(&temp_path, content)?;

    // Atomic rename
    fs::rename(temp_path, depfile_path)?;

    Ok(())
}

/// Get the output files that will be created for detection
///
/// **DEPRECATED**: Use OutputManager::get_produced_outputs() instead.
/// This function duplicates OutputManager logic and can get out of sync.
#[deprecated(
    note = "Use OutputManager::get_produced_outputs() instead to avoid synchronization issues"
)]
pub fn get_detection_output_files(
    input_path: &Path,
    crop_classes: &std::collections::HashSet<crate::config::DetectionClass>,
    bounding_box: bool,
    metadata: bool,
    output_dir: Option<&str>,
) -> Vec<PathBuf> {
    let mut outputs = Vec::new();

    let input_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let base_dir = if let Some(output_dir) = output_dir {
        PathBuf::from(output_dir)
    } else {
        input_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    // Add crop outputs - detection uses "crop" suffix regardless of class
    if !crop_classes.is_empty() {
        let crop_filename = format!("{input_stem}_crop.jpg");
        outputs.push(base_dir.join(crop_filename));
    }

    if bounding_box {
        let bbox_filename = format!("{input_stem}_bounding-box.jpg");
        outputs.push(base_dir.join(bbox_filename));
    }

    if metadata {
        let metadata_filename = format!("{input_stem}.beaker.toml");
        outputs.push(base_dir.join(metadata_filename));
    }

    outputs
}

/// Get the output files that will be created for cutout processing
///
/// **DEPRECATED**: Use OutputManager::get_produced_outputs() instead.
/// This function duplicates OutputManager logic and can get out of sync.
#[deprecated(
    note = "Use OutputManager::get_produced_outputs() instead to avoid synchronization issues"
)]
pub fn get_cutout_output_files(
    input_path: &Path,
    save_mask: bool,
    metadata: bool,
    output_dir: Option<&str>,
) -> Vec<PathBuf> {
    let mut outputs = Vec::new();

    let input_stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");

    let base_dir = if let Some(output_dir) = output_dir {
        PathBuf::from(output_dir)
    } else {
        input_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    // Always produces cutout
    let cutout_filename = format!("{input_stem}_cutout.png");
    outputs.push(base_dir.join(cutout_filename));

    if save_mask {
        let mask_filename = format!("{input_stem}_mask.png");
        outputs.push(base_dir.join(mask_filename));
    }

    if metadata {
        let metadata_filename = format!("{input_stem}.beaker.toml");
        outputs.push(base_dir.join(metadata_filename));
    }

    outputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn test_path_escaping() {
        // Test normal path
        let normal_path = Path::new("/path/to/file.txt");
        assert_eq!(escape_path_for_make(normal_path), "/path/to/file.txt");

        // Test path with spaces
        let space_path = Path::new("/path/with spaces/file.txt");
        assert_eq!(
            escape_path_for_make(space_path),
            "/path/with\\ spaces/file.txt"
        );

        // Test path with dollar signs
        let dollar_path = Path::new("/path/$VAR/file.txt");
        assert_eq!(escape_path_for_make(dollar_path), "/path/$$VAR/file.txt");
    }

    #[test]
    fn test_depfile_content_generation() {
        let targets = vec![PathBuf::from("output.png"), PathBuf::from("output.toml")];
        let prerequisites = vec![
            PathBuf::from("input.jpg"),
            PathBuf::from("/cache/stamp1.stamp"),
        ];

        let content = generate_depfile_content(&targets, &prerequisites).unwrap();
        let expected = "output.png output.toml: input.jpg /cache/stamp1.stamp\n";
        assert_eq!(content, expected);
    }

    #[test]
    fn test_depfile_content_with_spaces() {
        let targets = vec![PathBuf::from("output with spaces.png")];
        let prerequisites = vec![PathBuf::from("input with spaces.jpg")];

        let content = generate_depfile_content(&targets, &prerequisites).unwrap();
        let expected = "output\\ with\\ spaces.png: input\\ with\\ spaces.jpg\n";
        assert_eq!(content, expected);
    }

    #[test]
    fn test_detection_output_files() {
        use crate::config::DetectionClass;
        use std::collections::HashSet;

        let input_path = Path::new("/input/bird.jpg");

        // Test with head crop and metadata
        let mut crop_classes = HashSet::new();
        crop_classes.insert(DetectionClass::Head);
        let outputs = get_detection_output_files(&input_path, &crop_classes, false, true, None);
        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|p| p.file_name().unwrap() == "bird_crop.jpg"));
        assert!(outputs
            .iter()
            .any(|p| p.file_name().unwrap() == "bird.beaker.toml"));

        // Test with output directory
        let outputs =
            get_detection_output_files(&input_path, &crop_classes, false, false, Some("/output"));
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], PathBuf::from("/output/bird_crop.jpg"));
    }

    #[test]
    fn test_cutout_output_files() {
        let input_path = Path::new("/input/photo.jpg");

        // Test basic cutout
        let outputs = get_cutout_output_files(input_path, false, false, None);
        assert_eq!(outputs.len(), 1);
        assert!(outputs
            .iter()
            .any(|p| p.file_name().unwrap() == "photo_cutout.png"));

        // Test with mask and metadata
        let outputs = get_cutout_output_files(input_path, true, true, None);
        assert_eq!(outputs.len(), 3);
        assert!(outputs
            .iter()
            .any(|p| p.file_name().unwrap() == "photo_cutout.png"));
        assert!(outputs
            .iter()
            .any(|p| p.file_name().unwrap() == "photo_mask.png"));
        assert!(outputs
            .iter()
            .any(|p| p.file_name().unwrap() == "photo.beaker.toml"));
    }

    #[test]
    fn test_atomic_depfile_write() {
        let temp_dir = tempdir().unwrap();
        let depfile_path = temp_dir.path().join("test.d");
        let content = "target: prereq1 prereq2\n".to_string();

        write_depfile_atomically(&depfile_path, &content).unwrap();

        assert!(depfile_path.exists());
        assert_eq!(fs::read_to_string(&depfile_path).unwrap(), content);
    }
}
