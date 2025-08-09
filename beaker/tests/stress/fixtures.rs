//! Test fixtures with embedded models for deterministic stress testing

/// Test fixtures containing embedded models and their checksums
pub struct TestFixtures {
    pub small_model_bytes: &'static [u8],
    pub small_model_sha256: &'static str,
    pub corrupted_model_bytes: Vec<u8>,
    pub large_model_bytes: &'static [u8],
    pub large_model_sha256: &'static str,
}

impl TestFixtures {
    pub fn new() -> Self {
        // Create a simple deterministic "ONNX" model for testing
        // This is just test data, not a real ONNX model
        let small_model_data = b"FAKE_ONNX_MODEL_FOR_TESTING_PURPOSES_SMALL";
        let large_model_data = vec![0x42; 2048]; // 2KB of 0x42 bytes

        // Calculate SHA-256 checksums
        let small_model_sha256 = calculate_sha256_bytes(small_model_data);
        let large_model_sha256 = calculate_sha256_bytes(&large_model_data);

        // Create corrupted variant by flipping first byte
        let mut corrupted_model = small_model_data.to_vec();
        if !corrupted_model.is_empty() {
            corrupted_model[0] = !corrupted_model[0];
        }

        Self {
            small_model_bytes: small_model_data,
            small_model_sha256: Box::leak(small_model_sha256.into_boxed_str()),
            corrupted_model_bytes: corrupted_model,
            large_model_bytes: Box::leak(large_model_data.into_boxed_slice()),
            large_model_sha256: Box::leak(large_model_sha256.into_boxed_str()),
        }
    }
}

impl Default for TestFixtures {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate SHA-256 hash of bytes
fn calculate_sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixtures_creation() {
        let fixtures = TestFixtures::new();

        // Verify small model has content
        assert!(!fixtures.small_model_bytes.is_empty());
        assert!(!fixtures.small_model_sha256.is_empty());

        // Verify corrupted model is different from original
        assert_ne!(
            fixtures.small_model_bytes.to_vec(),
            fixtures.corrupted_model_bytes
        );

        // Verify large model has expected size
        assert_eq!(fixtures.large_model_bytes.len(), 2048);

        // Verify SHA-256 checksums are valid hex strings
        assert!(fixtures
            .small_model_sha256
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
        assert!(fixtures
            .large_model_sha256
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
        assert_eq!(fixtures.small_model_sha256.len(), 64); // SHA-256 is 64 hex chars
        assert_eq!(fixtures.large_model_sha256.len(), 64);
    }

    #[test]
    fn test_checksum_verification() {
        let fixtures = TestFixtures::new();

        // Verify that checksums match the actual data
        let actual_small_checksum = calculate_sha256_bytes(fixtures.small_model_bytes);
        assert_eq!(actual_small_checksum, fixtures.small_model_sha256);

        let actual_large_checksum = calculate_sha256_bytes(fixtures.large_model_bytes);
        assert_eq!(actual_large_checksum, fixtures.large_model_sha256);

        // Verify corrupted data has different checksum
        let corrupted_checksum = calculate_sha256_bytes(&fixtures.corrupted_model_bytes);
        assert_ne!(corrupted_checksum, fixtures.small_model_sha256);
    }
}
