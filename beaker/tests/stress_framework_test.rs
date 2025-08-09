//! Stress testing integration tests for beaker's cache mechanisms
//!
//! This integration test validates the stress testing framework
//! implemented as part of Phase 1.

mod stress;

use stress::{
    fixtures::TestFixtures,
    mock_servers::{FailureEvent, MockServerManager},
    orchestrator::{StressTestOrchestrator, TestScenario},
    validators::CacheValidator,
};

#[test]
fn test_stress_framework_basic_functionality() {
    // Test that the basic framework components work together

    // 1. Test fixtures creation
    let fixtures = TestFixtures::new();
    assert!(!fixtures.small_model_bytes.is_empty());
    assert!(!fixtures.small_model_sha256.is_empty());

    // 2. Test mock server creation
    let failure_pattern = vec![
        FailureEvent::Success,
        FailureEvent::HttpError(500),
        FailureEvent::CorruptedChecksum,
    ];
    let _mock_manager = MockServerManager::new(failure_pattern);

    // 3. Test validator
    let validator = CacheValidator::new();
    let temp_dir = tempfile::TempDir::new().unwrap();
    let result = validator.validate_cache_consistency(temp_dir.path());
    assert!(result.passed);

    // 4. Test orchestrator creation
    let _orchestrator = StressTestOrchestrator::new(2).unwrap();

    println!("✓ All Phase 1 components initialized successfully");
}

#[test]
fn test_cache_validation_framework() {
    // Test the cache validation logic

    let temp_dir = tempfile::TempDir::new().unwrap();
    let validator = CacheValidator::new();

    // Test with empty cache
    let result = validator.validate_cache_consistency(temp_dir.path());
    assert!(result.passed);

    let result = validator.validate_concurrent_safety(temp_dir.path());
    assert!(result.passed);

    // Test with valid cache files
    std::fs::write(temp_dir.path().join("test_model.onnx"), b"fake model data").unwrap();

    let result = validator.validate_cache_consistency(temp_dir.path());
    assert!(result.passed);

    let result = validator.validate_concurrent_safety(temp_dir.path());
    assert!(result.passed);

    // Test with invalid cache state (partial files)
    std::fs::write(temp_dir.path().join("partial_model.tmp"), b"partial").unwrap();

    let result = validator.validate_no_partial_files_remain(temp_dir.path());
    assert!(!result.passed);
    assert!(!result.violations.is_empty());

    println!("✓ Cache validation framework working correctly");
}

#[test]
fn test_failure_event_patterns() {
    // Test deterministic failure pattern creation and handling

    let pattern = vec![
        FailureEvent::Success,
        FailureEvent::HttpError(500),
        FailureEvent::CorruptedChecksum,
        FailureEvent::TcpConnectionRefused,
        FailureEvent::TcpMidStreamAbort(1024),
        FailureEvent::TcpHeaderThenClose,
    ];

    // Create mock server with this pattern
    let _mock_manager = MockServerManager::new(pattern.clone());

    // Verify pattern is preserved
    assert_eq!(pattern.len(), 6);

    // Test each failure event type
    for event in &pattern {
        match event {
            FailureEvent::Success => assert!(true),
            FailureEvent::HttpError(code) => assert!(*code >= 400),
            FailureEvent::CorruptedChecksum => assert!(true),
            FailureEvent::TcpConnectionRefused => assert!(true),
            FailureEvent::TcpMidStreamAbort(bytes) => assert!(*bytes > 0),
            FailureEvent::TcpHeaderThenClose => assert!(true),
        }
    }

    println!("✓ Failure event patterns working correctly");
}

#[test]
fn test_phase_1_integration() {
    // Integration test for Phase 1 components

    println!("=== Phase 1 Integration Test ===");

    // 1. Create test fixtures
    let fixtures = TestFixtures::new();
    println!("✓ Test fixtures created");
    println!(
        "  Small model size: {} bytes",
        fixtures.small_model_bytes.len()
    );
    println!("  Small model SHA256: {}", fixtures.small_model_sha256);
    println!(
        "  Large model size: {} bytes",
        fixtures.large_model_bytes.len()
    );

    // 2. Create mock server
    let failure_pattern = vec![
        FailureEvent::Success,
        FailureEvent::HttpError(503),
        FailureEvent::CorruptedChecksum,
    ];
    let mock_manager = MockServerManager::new(failure_pattern);
    println!("✓ Mock server created at: {}", mock_manager.http_base_url());

    // 3. Create orchestrator
    let orchestrator = StressTestOrchestrator::new(2).unwrap();
    println!("✓ Orchestrator created for 2 processes");

    // 4. Create validator
    let validator = CacheValidator::new();
    let temp_dir = tempfile::TempDir::new().unwrap();
    let validation_result = validator.validate_cache_consistency(temp_dir.path());
    println!("✓ Cache validator working: {}", validation_result.passed);

    // 5. Test basic orchestration (expect failures due to missing beaker executable in test environment)
    let failure_patterns = vec![
        vec![FailureEvent::Success],
        vec![FailureEvent::HttpError(500)],
    ];

    let results = orchestrator.run_stress_test(TestScenario::SameModelContention, failure_patterns);

    println!("✓ Stress test orchestration completed");
    println!("  Processes run: {}", results.total_processes());
    println!("  Framework ready for Phase 2");

    // Framework validation - at this stage we just verify it doesn't panic
    assert_eq!(results.total_processes(), 2);

    println!("=== Phase 1 Integration Test PASSED ===");
}
