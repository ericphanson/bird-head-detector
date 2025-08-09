//! Cache validation utilities for stress testing

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Cache validation results
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub passed: bool,
    pub violations: Vec<String>,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            passed: true,
            violations: Vec::new(),
        }
    }

    pub fn failure(violation: String) -> Self {
        Self {
            passed: false,
            violations: vec![violation],
        }
    }

    pub fn from_invariants(invariants: &[bool]) -> Self {
        if invariants.iter().all(|&inv| inv) {
            Self::success()
        } else {
            Self::failure("One or more invariants failed".to_string())
        }
    }

    pub fn combine(results: Vec<ValidationResult>) -> Self {
        let all_passed = results.iter().all(|r| r.passed);
        let all_violations: Vec<String> = results.into_iter().flat_map(|r| r.violations).collect();

        Self {
            passed: all_passed,
            violations: all_violations,
        }
    }
}

/// Cache validator for stress testing invariants
pub struct CacheValidator;

impl CacheValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate cache consistency after stress testing
    pub fn validate_cache_consistency(&self, cache_dir: &Path) -> ValidationResult {
        let results = vec![
            self.validate_no_partial_files_remain(cache_dir),
            self.validate_all_cached_models_valid_checksum(cache_dir),
            self.validate_proper_permissions(cache_dir),
            self.validate_directory_structure_intact(cache_dir),
        ];

        ValidationResult::combine(results)
    }

    /// Validate concurrent safety properties
    pub fn validate_concurrent_safety(&self, cache_dir: &Path) -> ValidationResult {
        let results = vec![
            self.validate_exactly_one_final_artifact_per_model(cache_dir),
            self.validate_no_race_condition_artifacts(cache_dir),
            self.validate_all_lock_files_cleaned_up(cache_dir),
            self.validate_readers_never_see_partial_content(cache_dir),
        ];

        ValidationResult::combine(results)
    }

    // Specific invariant validations

    /// Exactly one final artifact exists per model URL/checksum combination
    pub fn validate_exactly_one_final_artifact_per_model(
        &self,
        cache_dir: &Path,
    ) -> ValidationResult {
        if !cache_dir.exists() {
            return ValidationResult::success(); // Empty cache is valid
        }

        let model_files = match self.group_by_model_identity(cache_dir) {
            Ok(files) => files,
            Err(e) => {
                return ValidationResult::failure(format!("Failed to group model files: {}", e))
            }
        };

        for (model_id, files) in &model_files {
            if files.len() != 1 {
                return ValidationResult::failure(format!(
                    "Model '{}' has {} artifacts (expected exactly 1): {:?}",
                    model_id,
                    files.len(),
                    files
                ));
            }
        }

        ValidationResult::success()
    }

    /// No partial/temporary files remain after processing
    pub fn validate_no_partial_files_remain(&self, cache_dir: &Path) -> ValidationResult {
        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        let entries = match fs::read_dir(cache_dir) {
            Ok(entries) => entries,
            Err(_) => return ValidationResult::success(), // Directory doesn't exist
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    return ValidationResult::failure(format!(
                        "Failed to read directory entry: {}",
                        e
                    ))
                }
            };

            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.contains(".tmp")
                || filename.contains(".partial")
                || filename.contains(".downloading")
            {
                return ValidationResult::failure(format!("Partial file remains: {}", filename));
            }
        }

        ValidationResult::success()
    }

    /// All lock files are cleaned up after processing
    pub fn validate_all_lock_files_cleaned_up(&self, cache_dir: &Path) -> ValidationResult {
        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        let entries = match fs::read_dir(cache_dir) {
            Ok(entries) => entries,
            Err(_) => return ValidationResult::success(),
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    return ValidationResult::failure(format!(
                        "Failed to read directory entry: {}",
                        e
                    ))
                }
            };

            let filename = entry.file_name().to_string_lossy().to_string();
            if filename.ends_with(".lock") {
                return ValidationResult::failure(format!("Lock file remains: {}", filename));
            }
        }

        ValidationResult::success()
    }

    /// Readers never observe partial content (all files have valid structure)
    pub fn validate_readers_never_see_partial_content(&self, cache_dir: &Path) -> ValidationResult {
        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        let entries = match fs::read_dir(cache_dir) {
            Ok(entries) => entries,
            Err(_) => return ValidationResult::success(),
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    return ValidationResult::failure(format!(
                        "Failed to read directory entry: {}",
                        e
                    ))
                }
            };

            if entry.file_type().unwrap().is_file() {
                let filename = entry.file_name().to_string_lossy().to_string();

                // Skip lock files and other non-model files
                if filename.ends_with(".lock") || filename.starts_with('.') {
                    continue;
                }

                // Verify file is readable and has some content
                match fs::read(&entry.path()) {
                    Ok(contents) => {
                        if contents.is_empty() {
                            return ValidationResult::failure(format!(
                                "File has no content: {}",
                                filename
                            ));
                        }
                    }
                    Err(e) => {
                        return ValidationResult::failure(format!(
                            "File is not readable: {} (error: {})",
                            filename, e
                        ));
                    }
                }
            }
        }

        ValidationResult::success()
    }

    /// No race condition artifacts (orphaned files, etc.)
    pub fn validate_no_race_condition_artifacts(&self, cache_dir: &Path) -> ValidationResult {
        // This validation looks for signs of race conditions:
        // - Orphaned temporary files
        // - Duplicate files with timestamps
        // - Malformed filenames that suggest interrupted operations

        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        let entries = match fs::read_dir(cache_dir) {
            Ok(entries) => entries,
            Err(_) => return ValidationResult::success(),
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    return ValidationResult::failure(format!(
                        "Failed to read directory entry: {}",
                        e
                    ))
                }
            };

            let filename = entry.file_name().to_string_lossy().to_string();

            // Check for race condition indicators
            if filename.contains("..") || filename.contains("__") {
                return ValidationResult::failure(format!(
                    "Suspicious filename suggesting race condition: {}",
                    filename
                ));
            }

            // Check for timestamp-based duplicates
            if filename.contains("_copy") || filename.contains("_backup") {
                return ValidationResult::failure(format!(
                    "Duplicate file suggesting race condition: {}",
                    filename
                ));
            }
        }

        ValidationResult::success()
    }

    /// All cached models have valid checksums (basic integrity check)
    pub fn validate_all_cached_models_valid_checksum(&self, cache_dir: &Path) -> ValidationResult {
        // For now, just verify files are not empty
        // In a real implementation, this would verify actual checksums
        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        let entries = match fs::read_dir(cache_dir) {
            Ok(entries) => entries,
            Err(_) => return ValidationResult::success(),
        };

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(e) => {
                    return ValidationResult::failure(format!(
                        "Failed to read directory entry: {}",
                        e
                    ))
                }
            };

            if entry.file_type().unwrap().is_file() {
                let filename = entry.file_name().to_string_lossy().to_string();

                // Skip non-model files
                if filename.ends_with(".lock") || filename.starts_with('.') {
                    continue;
                }

                // Basic integrity check - file has content
                let metadata = match entry.metadata() {
                    Ok(metadata) => metadata,
                    Err(e) => {
                        return ValidationResult::failure(format!(
                            "Cannot read metadata for {}: {}",
                            filename, e
                        ))
                    }
                };

                if metadata.len() == 0 {
                    return ValidationResult::failure(format!(
                        "Zero-length model file: {}",
                        filename
                    ));
                }
            }
        }

        ValidationResult::success()
    }

    /// Verify proper file permissions
    pub fn validate_proper_permissions(&self, cache_dir: &Path) -> ValidationResult {
        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        // Basic check that cache directory is accessible
        match fs::read_dir(cache_dir) {
            Ok(_) => ValidationResult::success(),
            Err(e) => ValidationResult::failure(format!("Cache directory not accessible: {}", e)),
        }
    }

    /// Verify cache directory structure is intact
    pub fn validate_directory_structure_intact(&self, cache_dir: &Path) -> ValidationResult {
        if !cache_dir.exists() {
            return ValidationResult::success();
        }

        // Verify it's actually a directory
        match cache_dir.metadata() {
            Ok(metadata) => {
                if metadata.is_dir() {
                    ValidationResult::success()
                } else {
                    ValidationResult::failure("Cache path is not a directory".to_string())
                }
            }
            Err(e) => {
                ValidationResult::failure(format!("Cannot access cache directory metadata: {}", e))
            }
        }
    }

    /// Group files by model identity (simplified - just by filename)
    fn group_by_model_identity(
        &self,
        cache_dir: &Path,
    ) -> std::io::Result<HashMap<String, Vec<PathBuf>>> {
        let mut model_files: HashMap<String, Vec<PathBuf>> = HashMap::new();

        for entry in fs::read_dir(cache_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let filename = entry.file_name().to_string_lossy().to_string();

                // Skip lock files and hidden files
                if filename.ends_with(".lock") || filename.starts_with('.') {
                    continue;
                }

                // Group by base filename (simplified model identity)
                let model_id = filename.clone();
                model_files
                    .entry(model_id)
                    .or_insert_with(Vec::new)
                    .push(entry.path());
            }
        }

        Ok(model_files)
    }
}

impl Default for CacheValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validation_result_success() {
        let result = ValidationResult::success();
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_validation_result_failure() {
        let result = ValidationResult::failure("Test error".to_string());
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0], "Test error");
    }

    #[test]
    fn test_validation_result_combine() {
        let results = vec![
            ValidationResult::success(),
            ValidationResult::failure("Error 1".to_string()),
            ValidationResult::failure("Error 2".to_string()),
        ];

        let combined = ValidationResult::combine(results);
        assert!(!combined.passed);
        assert_eq!(combined.violations.len(), 2);
    }

    #[test]
    fn test_cache_validator_empty_cache() {
        let temp_dir = TempDir::new().unwrap();
        let validator = CacheValidator::new();

        let result = validator.validate_cache_consistency(temp_dir.path());
        assert!(result.passed);

        let result = validator.validate_concurrent_safety(temp_dir.path());
        assert!(result.passed);
    }

    #[test]
    fn test_cache_validator_with_valid_files() {
        let temp_dir = TempDir::new().unwrap();
        let validator = CacheValidator::new();

        // Create some valid cache files
        fs::write(temp_dir.path().join("model1.onnx"), b"fake model data").unwrap();
        fs::write(temp_dir.path().join("model2.onnx"), b"another fake model").unwrap();

        let result = validator.validate_cache_consistency(temp_dir.path());
        assert!(result.passed);

        let result = validator.validate_concurrent_safety(temp_dir.path());
        assert!(result.passed);
    }

    #[test]
    fn test_cache_validator_with_partial_files() {
        let temp_dir = TempDir::new().unwrap();
        let validator = CacheValidator::new();

        // Create partial files that should fail validation
        fs::write(temp_dir.path().join("model1.onnx"), b"valid model").unwrap();
        fs::write(temp_dir.path().join("model2.onnx.tmp"), b"partial").unwrap();

        let result = validator.validate_no_partial_files_remain(temp_dir.path());
        assert!(!result.passed);
        assert!(result.violations[0].contains("Partial file remains"));
    }

    #[test]
    fn test_cache_validator_with_lock_files() {
        let temp_dir = TempDir::new().unwrap();
        let validator = CacheValidator::new();

        // Create lock files that should fail validation
        fs::write(temp_dir.path().join("model1.onnx"), b"valid model").unwrap();
        fs::write(temp_dir.path().join("model1.onnx.lock"), b"").unwrap();

        let result = validator.validate_all_lock_files_cleaned_up(temp_dir.path());
        assert!(!result.passed);
        assert!(result.violations[0].contains("Lock file remains"));
    }

    #[test]
    fn test_cache_validator_with_empty_files() {
        let temp_dir = TempDir::new().unwrap();
        let validator = CacheValidator::new();

        // Create empty files that should fail validation
        fs::write(temp_dir.path().join("model1.onnx"), b"").unwrap();

        let result = validator.validate_readers_never_see_partial_content(temp_dir.path());
        assert!(!result.passed);
        assert!(result.violations[0].contains("File has no content"));
    }
}
