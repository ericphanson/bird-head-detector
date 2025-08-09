# Beaker Test Framework

End-to-end integration tests for the beaker CLI focusing on metadata validation, performance tracking, and stress testing. These tests exercise the complete CLI tool to verify it "really works" while remaining independent of implementation details.

## Goals

- Validate metadata output across device configurations and processing modes
- Track performance characteristics to identify regressions
- Ensure consistent behavior across head detection and cutout operations
- Provide parallel test execution for faster feedback
- **[NEW]** Stress test caching mechanisms under concurrent access and network failures

## Test Architecture

The test framework is split across multiple components for clarity and maintainability:

### Core Integration Tests

#### `metadata_based_tests.rs` - Test Scenarios
Contains the actual test definitions in `get_test_scenarios()` and the macro call to generate tests. Start here when adding new test cases or understanding what scenarios are covered.

#### `metadata_test_framework.rs` - Framework Code
Houses the testing infrastructure: validation logic, metadata parsing, command execution, and test generation macro. Edit this when adding new validation types or framework features.

#### `test_performance_tracker.rs` - Performance Monitoring
Handles timing collection and reporting. Modify this to adjust performance thresholds or add new metrics.

### Stress Testing Framework

#### `PARALLEL_STRESS_TEST_PLAN.md` - Stress Test Plan
**Comprehensive plan for parallel process stress testing** focusing on:
- ONNX and CoreML cache robustness under concurrent access
- Network failure simulation and recovery testing
- Lock contention and race condition detection
- Cache consistency validation across multiple processes

See the plan for detailed framework architecture, implementation strategy, and test scenarios for validating cache mechanisms under realistic concurrent usage patterns.

### Metadata-Based Testing
Tests use scenario definitions with expected metadata validation checks. Each scenario exercises specific combinations of tools, devices, and configurations while verifying the generated `.beaker.toml` metadata.

### Performance Tracking
Automatic timing collection for wall-clock and CPU time, with thresholds to flag performance issues during development.

## Running Tests

```bash
# Run metadata tests with performance output
cargo test --test metadata_based_tests -- --nocapture

# View performance summary (requires --nocapture flag)
# Reports wall-clock time vs CPU time for parallelization efficiency
```

## Performance Output

After running metadata tests with `--nocapture`, you'll see:
- Total wall-clock time vs CPU time
- Head/cutout operation counts
- Slowest individual tests
- Warnings for tests exceeding 30s

## Adding Test Scenarios

1. Define scenario in `get_test_scenarios()` function in `metadata_based_tests.rs`:

```rust
TestScenario {
    name: "your_scenario_name",
    tool: "head", // or "cutout" or "both"
    args: vec!["--your-arg", "value"],
    expected_files: vec!["output.jpg", "output.beaker.toml"],
    metadata_checks: vec![
        MetadataCheck::DeviceUsed("head", "auto"),
        MetadataCheck::FilesProcessed("head", 1),
        // Add validation checks as needed
    ],
}
```

2. Add scenario name to `generate_metadata_tests!` macro in the same file
3. Automatic validation ensures no scenarios are missed

## Metadata Validation Checks

The framework provides several validation checks for `.beaker.toml` metadata:

### Available Checks

- **`DeviceUsed(tool, device)`** - Verifies which device was used (e.g., "cpu", "auto", "cuda")
- **`FilesProcessed(tool, count)`** - Confirms number of files processed by the tool
- **`ConfigValue(tool, field_path, expected_value)`** - Validates specific configuration values using dot notation
- **`TimingBound(tool, field, min_ms, max_ms)`** - Ensures timing fields fall within expected ranges
- **`OutputCreated(filename)`** - Confirms expected output files were created
- **`ExecutionProvider(tool, provider)`** - Verifies execution provider (e.g., "onnxruntime")
- **`ExitCode(tool, expected_code)`** - Validates tool exit codes
- **`BeakerVersion(tool)`** - Ensures beaker version is recorded in metadata
- **`CoreResultsField(tool, field_name)`** - Checks for presence of core result fields

### Adding New Checks

1. Add a new variant to the `MetadataCheck` enum in `metadata_test_framework.rs`
2. Implement validation logic in the `validate_metadata_check()` function's match statement
3. Use the new check in test scenarios in `metadata_based_tests.rs`

Example:
```rust
// In MetadataCheck enum (metadata_test_framework.rs)
NewValidation(&'static str, &'static str), // tool, expected_value

// In validate_metadata_check() function (metadata_test_framework.rs)
MetadataCheck::NewValidation(tool, expected) => {
    // Add validation logic here
}
```
