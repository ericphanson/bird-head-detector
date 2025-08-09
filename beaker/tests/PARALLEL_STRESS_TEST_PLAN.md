# Parallel Process Stress Test Plan

## Overview

This document outlines a comprehensive testing framework for validating the robustness of beaker's ONNX and CoreML caches under concurrent access with simulated network failures. The framework will ensure cache controls work correctly when multiple beaker processes execute simultaneously in noisy network environments.

## Background

Beaker uses two primary caching mechanisms:
1. **ONNX Model Cache**: Downloads and caches ONNX models with SHA-256 verification and lock-file based concurrency protection
2. **CoreML Cache**: On Apple Silicon, ONNX Runtime compiles ONNX models to CoreML format and caches the result

Both caches must handle concurrent access gracefully without corruption, race conditions, or deadlocks.

## Goals

- **Reliability**: Ensure caches are robust to concurrent access patterns
- **Failure Resilience**: Validate graceful handling of network failures, corruption, and connection issues
- **Deterministic**: Event-based testing with eventual invariants - no timing dependencies
- **Fast Tests**: Virtualized time and immediate failure injection for rapid CI execution
- **No Flakiness**: Logic-based validation that always converges to correct state

## Framework Architecture

### 1. Crate Analysis and Selection

#### HTTP Mocking Options Evaluated

**httpmock = "0.7.0"** (PRIMARY)
- ✅ Simpler API focused on HTTP mocking
- ✅ No async complexity - works with beaker's blocking model
- ✅ MIT/Apache-2.0 license
- ✅ Good for simulating HTTP-level failures and responses
- ✅ Thread-safe for concurrent test execution
- ✅ Lighter weight than wiremock

**Small hyper helper** (COMPLEMENTARY)
- ✅ Enables true TCP-level failures that httpmock cannot simulate
- ✅ Mid-stream socket abort and connection refused scenarios
- ✅ ~40-60 LOC, stays within cargo test framework
- ✅ No docker, no external dependencies
- ✅ Covers highest-value network fault types

**wiremock = "0.6.4"** (Alternative)
- ⚠️ Adds async complexity with tokio dependency
- ⚠️ More heavyweight for simple HTTP mocking needs
- ✅ More features but potentially overkill

#### Stress Testing Utilities

**cargo-stress = "0.2.0"**
- ✅ Specifically designed for catching non-deterministic failures
- ✅ Could complement our framework for test reliability
- ⚠️ Focused on existing test discovery, not custom stress scenarios

#### Process Management

**std::process::Command** (Current approach)
- ✅ Already used in existing test framework
- ✅ Platform independent
- ✅ Full control over environment and arguments
- ✅ No additional dependencies

### 2. Mock HTTP Server with Failure Injection

**Primary Tool**: `httpmock = "0.7.0"` for HTTP-level failures
**Supplementary Tool**: Small `hyper` helper for TCP-level failures

**HTTP-Level Failure Injection** (via httpmock):
- **HTTP Errors**: Return 4xx/5xx status codes at predetermined points
- **Corrupted Data**: Serve models with incorrect SHA-256 checksums (deterministic corruption)
- **Invalid Headers**: Malformed content-length or missing headers
- **Deterministic Patterns**: Predefined failure sequences for reproducible testing

**TCP-Level Failure Injection** (via hyper helper):
- **Connection Refused**: No server listening on port (not achievable via HTTP status)
- **Mid-Stream Abort**: Write headers + N bytes, then close socket abruptly
- **Header-then-Stall**: Write headers, then close immediately without body
- **Immediate Close**: Accept connection, then close without any response

**Small Hyper Helper Implementation** (~50 LOC):
```rust
// tests/stress/tcp_fault_server.rs
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

pub enum TcpFaultType {
    Success(Vec<u8>),           // Normal response
    MidStreamAbort(usize),      // Close after N bytes
    HeaderThenClose,            // Headers only, then close
    ImmediateClose,             // Accept then close
}

pub struct TcpFaultServer {
    fault_sequence: Vec<TcpFaultType>,
    request_counter: AtomicUsize,
}

impl TcpFaultServer {
    pub fn start(faults: Vec<TcpFaultType>) -> (String, tokio::task::JoinHandle<()>) {
        // Start server on random port, return URL and handle
        // Implement fault injection based on request counter
    }
}
```

### 3. Test Fixtures and Model Preparation

**Embedded Test Fixtures** (replaces real model downloads):
```rust
// tests/fixtures/models.rs
pub struct TestFixtures {
    pub small_model_bytes: &'static [u8],
    pub small_model_sha256: &'static str,
    pub corrupted_model_bytes: &'static [u8],  // First byte flipped
    pub large_model_bytes: &'static [u8],      // For partial download tests
    pub large_model_sha256: &'static str,
}

impl TestFixtures {
    pub const fn new() -> Self {
        Self {
            // Small deterministic model for basic tests (~200-500 KB)
            small_model_bytes: include_bytes!("small_test_model.onnx"),
            small_model_sha256: "a1b2c3d4e5f6...", // Precomputed SHA-256

            // Corrupted variant (first byte flipped)
            corrupted_model_bytes: {
                let mut data = *include_bytes!("small_test_model.onnx");
                data[0] = !data[0];  // Flip first bit
                &data
            },

            // Larger model for partial download stress (~1-2 MB)
            large_model_bytes: include_bytes!("large_test_model.onnx"),
            large_model_sha256: "f6e5d4c3b2a1...", // Precomputed SHA-256
        }
    }
}

// Generate fixtures if they don't exist
fn generate_test_fixtures() -> TestFixtures {
    // Create minimal valid ONNX files or use deterministic byte patterns
    // Include correct SHA-256 checksums
    // Store in tests/fixtures/ directory
}
```

**Fixture Generation Strategy**:
- Use `include_bytes!()` for zero runtime overhead
- Generate minimal valid ONNX files if needed (or deterministic byte patterns)
- Precompute SHA-256 checksums for integrity testing
- Multiple sizes: small (~200KB), large (~1-2MB) for different test scenarios
- Embedded corrupted variants for checksum mismatch testing

### 4. Concurrency Test Scenarios

#### Scenario 1: Concurrent Same-Model Downloads (Shared Cache)
**Purpose**: Test ONNX cache lock contention with real race conditions
```rust
fn test_concurrent_same_model_download() {
    // Launch 10 beaker processes simultaneously requesting same model
    // ALL processes use SHARED cache directory - this is critical for lock contention
    // Verify only one downloads, others wait for completion
    // Ensure all processes end up with valid cached model
    // Check lock file cleanup
}
```

#### Scenario 2: Concurrent Different-Model Downloads (Shared Cache)
**Purpose**: Test cache isolation and parallel downloads without interference
```rust
fn test_concurrent_different_model_downloads() {
    // Launch processes requesting different models simultaneously
    // ALL processes use SHARED cache directory for isolation testing
    // Verify parallel downloads work without interference
    // Check each model caches correctly with proper SHA-256 checksums
}
```

#### Scenario 3: Network Failure During Download (Shared Cache)
**Purpose**: Test failure recovery and cache consistency under contention
```rust
fn test_network_failure_recovery() {
    // Start downloads, inject TCP-level failures partway through
    // ALL processes use SHARED cache directory
    // Verify partial downloads are cleaned up
    // Ensure retry logic works correctly with lock contention
    // Check lock files don't become stale across processes
}
```

#### Scenario 4: Corrupted Download Handling (Shared Cache)
**Purpose**: Test SHA-256 validation under concurrent access
```rust
fn test_corrupted_download_handling() {
    // Serve model with wrong SHA-256 checksum to some processes
    // ALL processes use SHARED cache directory
    // Verify corrupted downloads are rejected and purged
    // Ensure cache doesn't get poisoned
    // Check concurrent processes handle failures independently
}
```

#### Scenario 5: Cache Directory Stress (Shared Cache)
**Purpose**: Test filesystem operations under heavy concurrency
```rust
fn test_cache_directory_stress() {
    // Multiple processes creating/accessing shared cache directory
    // Concurrent file creation, locking, and cleanup
    // Test permission and ownership handling
    // Verify cache cleanup operations under load
}
```

#### Scenario 6: Different-Model Isolation (Per-Process Cache)
**Purpose**: Test cache isolation when processes use separate cache directories
```rust
fn test_different_model_isolation() {
    // Each process uses its own cache directory
    // Verify no interference between isolated caches
    // Test for proper cache directory creation and management
}
```

#### Scenario 7: CoreML Cache Concurrency (macOS only, Shared Cache)
**Purpose**: Test CoreML compilation cache under load
```rust
#[cfg(target_os = "macos")]
fn test_coreml_cache_concurrency() {
    // Multiple processes compiling same ONNX model to CoreML
    // ALL processes use SHARED CoreML cache directory
    // Verify CoreML cache directory locking and compilation isolation
    // Check concurrent compilation handling
}
```

### 5. Deterministic Failure Injection

**Event-Based Failure Controller**:
```rust
struct FailureController {
    failure_sequence: Vec<FailureEvent>,
    current_request: AtomicUsize,
}

#[derive(Clone)]
enum FailureEvent {
    Success,
    HttpError(u16),              // HTTP status codes via httpmock
    CorruptedChecksum,           // Wrong SHA-256 via httpmock
    TcpConnectionRefused,        // Via hyper helper: no server on port
    TcpMidStreamAbort(usize),    // Via hyper helper: close after N bytes
    TcpHeaderThenClose,          // Via hyper helper: headers only
}

impl FailureController {
    fn new(pattern: Vec<FailureEvent>) -> Self {
        Self {
            failure_sequence: pattern,
            current_request: AtomicUsize::new(0),
        }
    }

    fn next_response(&self) -> FailureEvent {
        let index = self.current_request.fetch_add(1, Ordering::SeqCst);
        self.failure_sequence[index % self.failure_sequence.len()].clone()
    }
}
```

**Predefined Failure Patterns**:
- **TCP-Level Issues**: `[Success, TcpConnectionRefused, Success, TcpMidStreamAbort(1024)]`
- **HTTP-Level Issues**: `[Success, HttpError(500), CorruptedChecksum, Success]`
- **Mixed Failure Recovery**: `[TcpMidStreamAbort(512), HttpError(503), Success, Success]`
- **Checksum Corruption**: `[Success, CorruptedChecksum, Success, CorruptedChecksum]`

**Routing Between Mock Systems**:
```rust
fn configure_failure_injection(pattern: &[FailureEvent]) -> (MockServer, Option<TcpFaultServer>) {
    let mock_server = MockServer::start();
    let tcp_server = if pattern.iter().any(|e| matches!(e, FailureEvent::Tcp*)) {
        Some(TcpFaultServer::start(tcp_faults))
    } else {
        None
    };

    // Route HTTP-level failures to httpmock
    // Route TCP-level failures to hyper helper
    (mock_server, tcp_server)
}
```

### 6. Failpoints for Filesystem Race Conditions

**Failpoint Integration** (using `fail` crate in tests only):
```rust
// Critical points where corruption can occur during multi-process access
pub enum CacheFailpoint {
    AfterLockBeforeWrite,     // Process crash after acquiring lock
    AfterWriteBeforeRename,   // Process crash after write, before atomic rename
    AfterHashBeforeUnlock,    // Process crash after verification, before unlock
    DuringLockFileCleanup,    // Process crash during stale lock cleanup
}

fn inject_failpoint(point: CacheFailpoint, process_id: usize) {
    #[cfg(test)]
    {
        use fail::fail_point;
        let point_name = format!("cache_{}_{}", point.as_str(), process_id);
        fail_point!(&point_name);
    }
}
```

**Crash Recovery Test**:
```rust
fn test_crash_during_write_before_rename() {
    // Set up shared cache directory
    let shared_cache = TempDir::new().unwrap();

    // Configure failpoint for one process
    #[cfg(test)]
    fail::cfg("cache_after_write_before_rename_0", "panic").unwrap();

    // Start multiple processes downloading same model
    // Process 0 will crash after write but before atomic rename
    // Other processes should detect stale lock and recover

    // Validation:
    // - No orphaned .tmp files remain
    // - Exactly one valid final artifact exists
    // - All successful processes have consistent cache state
    // - Stale lock is cleaned up properly
}
```

**Cross-Process Lock Testing** (using `fd-lock` for validation):
```rust
fn test_cross_process_lock_semantics() {
    // Validate that file locks work correctly across processes
    // Test stale lock detection and cleanup
    // Verify lock acquisition/release patterns
    use fd_lock::RwLock;

    // Test that beaker's locking matches expected cross-process semantics
}
```

### 7. Process Management and Orchestration

**Event-Based Process Coordination**:
```rust
use std::sync::{Arc, Barrier, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};

struct StressTestOrchestrator {
    max_concurrent_processes: usize,
    failure_patterns: Vec<Vec<FailureEvent>>,
}

#[derive(Debug)]
struct ProcessResult {
    process_id: usize,
    exit_code: i32,
    cache_state: CacheValidationResult,
    error_output: String,
}

impl StressTestOrchestrator {
    fn run_stress_test(&self, scenario: TestScenario) -> StressTestResults {
        let start_barrier = Arc::new(Barrier::new(self.max_concurrent_processes + 1));
        let (result_tx, result_rx): (Sender<ProcessResult>, Receiver<ProcessResult>) = channel();

        // Launch processes with deterministic failure patterns
        let handles: Vec<_> = (0..self.max_concurrent_processes).map(|i| {
            let barrier = Arc::clone(&start_barrier);
            let tx = result_tx.clone();
            let failure_pattern = self.failure_patterns[i % self.failure_patterns.len()].clone();

            std::thread::spawn(move || {
                // Wait for all processes to be ready
                barrier.wait();

                // Execute beaker with deterministic environment
                let result = execute_beaker_process(i, &failure_pattern);
                tx.send(result).unwrap();
            })
        }).collect();

        // Start all processes simultaneously
        start_barrier.wait();

        // Collect results as they complete (no timeouts)
        let mut results = Vec::new();
        for _ in 0..self.max_concurrent_processes {
            results.push(result_rx.recv().unwrap());
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        StressTestResults::new(results)
    }
}
```

**Process Configuration**:
- **Shared Cache Testing**: Most scenarios use a single shared cache directory to test real lock contention and race conditions
- **Isolated Cache Testing**: Only specific isolation tests use per-process cache directories
- Environment variables control model URLs to point to appropriate mock servers (httpmock or hyper helper)
- Separate output directories to avoid conflicts
- Independent metadata file generation
- No timing dependencies - processes coordinate through barriers and channels

**Cache Directory Strategy**:
```rust
enum CacheStrategy {
    Shared(PathBuf),           // All processes use same cache dir (for contention testing)
    PerProcess(Vec<PathBuf>),  // Each process gets own cache dir (for isolation testing)
}

fn configure_cache_strategy(scenario: TestScenario) -> CacheStrategy {
    match scenario {
        TestScenario::SameModelContention => CacheStrategy::Shared(shared_cache_dir()),
        TestScenario::NetworkFailureRecovery => CacheStrategy::Shared(shared_cache_dir()),
        TestScenario::CorruptionHandling => CacheStrategy::Shared(shared_cache_dir()),
        TestScenario::CacheDirectoryStress => CacheStrategy::Shared(shared_cache_dir()),
        TestScenario::CoreMLConcurrency => CacheStrategy::Shared(shared_cache_dir()),
        TestScenario::DifferentModelIsolation => CacheStrategy::PerProcess(per_process_dirs()),
    }
}
```

### 8. Validation and Metrics Collection

**Cache State Validation**:
```rust
struct CacheValidator;

impl CacheValidator {
    fn validate_cache_consistency(&self, cache_dir: &Path) -> ValidationResult {
        // Check no partial/corrupted files remain
        // Verify all cached models have correct SHA-256 checksums
        // Ensure proper file permissions and ownership
        // Check cache directory structure integrity
        ValidationResult::from_invariants(&[
            self.no_partial_files_remain(cache_dir),
            self.all_cached_models_valid_checksum(cache_dir),
            self.proper_permissions(cache_dir),
            self.directory_structure_intact(cache_dir),
        ])
    }

    fn validate_concurrent_safety(&self, cache_dir: &Path) -> ValidationResult {
        // Validate shared cache state across all processes
        // Check for race condition artifacts
        // Validate lock file cleanup
        ValidationResult::from_invariants(&[
            self.exactly_one_final_artifact_per_model(cache_dir),
            self.no_race_condition_artifacts(cache_dir),
            self.all_lock_files_cleaned_up(cache_dir),
            self.readers_never_see_partial_content(cache_dir),
        ])
    }

    // Specific invariants that must hold eventually
    fn exactly_one_final_artifact_per_model(&self, cache_dir: &Path) -> bool {
        // For each model URL/checksum combination, exactly one final file exists
        // No duplicate or competing versions
        let model_files: HashMap<String, Vec<PathBuf>> = self.group_by_model_identity(cache_dir);
        model_files.values().all(|files| files.len() == 1)
    }

    fn no_partial_files_remain(&self, cache_dir: &Path) -> bool {
        !cache_dir.read_dir().unwrap()
            .any(|entry| {
                let name = entry.unwrap().file_name().to_string_lossy();
                name.contains(".tmp") || name.contains(".partial") || name.contains(".downloading")
            })
    }

    fn all_lock_files_cleaned_up(&self, cache_dir: &Path) -> bool {
        !cache_dir.read_dir().unwrap()
            .any(|entry| entry.unwrap().file_name().to_string_lossy().ends_with(".lock"))
    }

    fn readers_never_see_partial_content(&self, cache_dir: &Path) -> bool {
        // All readable files must have valid SHA-256 checksums
        // No process should ever read a half-written file
        cache_dir.read_dir().unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().unwrap().is_file())
            .filter(|entry| !entry.file_name().to_string_lossy().ends_with(".lock"))
            .all(|entry| self.file_has_valid_checksum(&entry.path()))
    }
}
```

**Logical Metrics Collection**:
- **Process Completion Patterns**: Success/failure counts with specific error types mapped to expected failure patterns
- **Cache Consistency Invariants**: Eventual properties validated after all processes complete
- **Lock File Lifecycle Tracking**: Acquire → release patterns with proper cleanup validation
- **Model Integrity Verification**: SHA-256 consistency across all successful processes
- **Error Recovery Analysis**: How deterministic failures propagate and resolve
- **Atomic Operation Validation**: Write → fsync → rename completeness under failpoint testing

**Metadata Analysis**:
- Leverage cache metadata functionality for stress test validation
- Track model load times under concurrent access
- Monitor cache hit/miss patterns
- Analyze failure recovery timing

### 9. Test Implementation Strategy

#### Phase 1: Foundation Infrastructure
1. Add httpmock and minimal hyper helper dependencies to `[dev-dependencies]`
2. Create `tests/stress/` module structure with embedded fixtures
3. Implement basic mock servers (httpmock + small hyper helper) with model serving
4. Build process orchestration framework with shared cache support
5. Add cache validation utilities with specific invariants

#### Phase 2: Basic Concurrency Tests with Shared Cache
1. Implement concurrent same-model download test (shared cache directory)
2. Add concurrent different-model download test (shared cache directory)
3. Create cache validation for these scenarios with new invariants
4. Ensure tests run reliably in CI environment under 30 seconds

#### Phase 3: TCP-Level Failure Injection
1. Add small hyper helper for connection refused and mid-stream abort
2. Implement corrupted download testing with SHA-256 validation
3. Create mixed HTTP/TCP failure pattern testing
4. Add failpoints around write → fsync → rename operations

#### Phase 4: Advanced Scenarios and Crash Recovery
1. Cache directory stress testing with shared cache
2. Failpoint-driven crash recovery testing
3. CoreML cache testing with shared cache (macOS)
4. Cross-process lock validation using fd-lock

#### Phase 5: Integration and CI
1. Integrate with existing test framework
2. Add CI job for stress testing (sub-5-minute target)
3. Performance regression detection via logical invariants
4. Documentation and maintenance guides

### 10. Implementation Example

Here's a concrete example of how the stress test framework would look:

```rust
// tests/stress/mod.rs
use httpmock::{MockServer, Mock, When, Then};
use std::process::Command;
use std::thread;
use std::sync::{Arc, Barrier, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::TempDir;

#[test]
fn test_concurrent_model_download_stress() {
    // Setup: Use embedded test fixtures instead of downloading
    let test_fixtures = TestFixtures::new();

    // Create deterministic failure pattern with TCP and HTTP failures
    let failure_pattern = vec![
        FailureEvent::Success,                      // Process 0: succeeds
        FailureEvent::TcpConnectionRefused,         // Process 1: TCP-level failure
        FailureEvent::TcpMidStreamAbort(1024),      // Process 2: partial TCP download
        FailureEvent::Success,                      // Process 3: succeeds
        FailureEvent::CorruptedChecksum,            // Process 4: gets corrupted data
    ];

    // Start mock servers with deterministic responses
    let mock_server = MockServer::start();
    let tcp_fault_server = TcpFaultServer::start(extract_tcp_faults(&failure_pattern));
    let failure_controller = Arc::new(FailureController::new(failure_pattern));

    // Configure mock responses - no delays, immediate responses
    let failure_controller_clone = Arc::clone(&failure_controller);
    Mock::new()
        .expect_request(When::path("/small-test-model.onnx"))
        .return_response_with(move |_| {
            match failure_controller_clone.next_response() {
                FailureEvent::Success => {
                    Then::new()
                        .status(200)
                        .body(test_fixtures.small_model_bytes.to_vec())
                }
                FailureEvent::HttpError(code) => {
                    Then::new().status(code)
                }
                FailureEvent::CorruptedChecksum => {
                    Then::new()
                        .status(200)
                        .body(test_fixtures.corrupted_model_bytes.to_vec())
                }
                // TCP-level failures handled by hyper helper, not httpmock
                FailureEvent::TcpConnectionRefused |
                FailureEvent::TcpMidStreamAbort(_) |
                FailureEvent::TcpHeaderThenClose => {
                    // Redirect to TCP fault server
                    Then::new()
                        .status(302)
                        .header("Location", &tcp_fault_server.url())
                }
            }
        })
        .create_on(&mock_server);

    // Environment setup for beaker processes - SHARED cache directory
    let temp_base = TempDir::new().unwrap();
    let shared_cache = temp_base.path().join("shared_cache");  // All processes use this
    std::fs::create_dir_all(&shared_cache).unwrap();
    let mock_url = format!("{}/small-test-model.onnx", mock_server.base_url());

    // Synchronization for simultaneous start
    let process_count = 5;
    let start_barrier = Arc::new(Barrier::new(process_count + 1));
    let (result_tx, result_rx): (Sender<ProcessResult>, Receiver<ProcessResult>) = channel();

    // Launch concurrent beaker processes
    let handles: Vec<_> = (0..process_count).map(|i| {
        let barrier = Arc::clone(&start_barrier);
        let shared_cache = shared_cache.clone();  // ALL processes use shared cache
        let output_dir = temp_base.path().join(format!("output_{}", i));
        let mock_url = mock_url.clone();
        let tx = result_tx.clone();

        thread::spawn(move || {
            // Wait for all processes to be ready
            barrier.wait();

            // Execute beaker process
            let result = Command::new("./target/debug/beaker")
                .args(&[
                    "cutout",
                    "../example.jpg",
                    "--metadata",
                    "--output-dir", output_dir.to_str().unwrap()
                ])
                .args(&["--model-url", &mock_url])
                .env("ONNX_MODEL_CACHE_DIR", shared_cache.to_str().unwrap())  // SHARED
                .output()
                .unwrap();

            let exit_code = result.status.code().unwrap_or(-1);
            let stderr = String::from_utf8_lossy(&result.stderr).to_string();

            tx.send(ProcessResult {
                process_id: i,
                exit_code,
                shared_cache_dir: shared_cache,  // Reference to shared cache
                output_dir: output_dir.clone(),
                error_output: stderr,
            }).unwrap();
        })
    }).collect();

    // Start all processes simultaneously
    start_barrier.wait();

    // Collect results as they complete (no timeouts)
    let mut results = Vec::new();
    for _ in 0..process_count {
        results.push(result_rx.recv().unwrap());
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Validation: Eventual invariants that must hold

    // At least some processes should succeed despite deterministic failures
    let success_count = results.iter()
        .filter(|r| r.exit_code == 0)
        .count();
    assert!(success_count >= 2, "Expected at least 2 successes, got {}", success_count);

    // Critical: Exactly one final artifact in shared cache for this model
    let cached_model = shared_cache.join("small-test-model.onnx");
    assert!(cached_model.exists(), "Shared cache should have the model");

    // Verify SHA-256 checksum on shared cache
    let checksum = calculate_sha256(&cached_model).unwrap();
    assert_eq!(checksum, test_fixtures.small_model_sha256);

    // All successful processes should reference the same cached file
    for result in &results {
        if result.exit_code == 0 {
            // Check metadata was generated
            let metadata_file = result.output_dir.join("example.beaker.toml");
            assert!(metadata_file.exists(), "Process {} should have metadata", result.process_id);
        }
    }

    // Critical invariant: No partial downloads or lock files remain in shared cache
    let lock_file = shared_cache.join("small-test-model.onnx.lock");
    assert!(!lock_file.exists(), "No lock files should remain in shared cache");

    // Check for partial/temporary files in shared cache
    let cache_entries: Vec<_> = std::fs::read_dir(&shared_cache)
        .unwrap()
        .collect();
    for entry in cache_entries {
        let entry = entry.unwrap();
        let filename = entry.file_name();
        assert!(!filename.to_string_lossy().contains(".tmp"),
               "No temporary files should remain: {:?}", filename);
        assert!(!filename.to_string_lossy().contains(".partial"),
               "No partial files should remain: {:?}", filename);
    }

    // Validate error patterns match expected TCP/HTTP failures
    let tcp_failures = results.iter()
        .filter(|r| r.exit_code != 0 && (
            r.error_output.contains("connection refused") ||
            r.error_output.contains("connection reset")
        ))
        .count();
    let checksum_failures = results.iter()
        .filter(|r| r.exit_code != 0 && r.error_output.contains("checksum"))
        .count();

    // Expect specific failure types based on our deterministic pattern
    assert!(tcp_failures >= 1, "Expected at least 1 TCP-level failure");
    assert!(checksum_failures >= 1, "Expected at least 1 SHA-256 checksum failure");
}

#[derive(Debug)]
struct ProcessResult {
    process_id: usize,
    exit_code: i32,
    shared_cache_dir: PathBuf,  // Reference to shared cache
    output_dir: PathBuf,
    error_output: String,
}

#[derive(Clone, Debug)]
enum FailureEvent {
    Success,
    HttpError(u16),              // HTTP status codes via httpmock
    CorruptedChecksum,           // Wrong SHA-256 via httpmock
    TcpConnectionRefused,        // Via hyper helper: no server on port
    TcpMidStreamAbort(usize),    // Via hyper helper: close after N bytes
    TcpHeaderThenClose,          // Via hyper helper: headers only
}

struct FailureController {
    failure_sequence: Vec<FailureEvent>,
    current_request: AtomicUsize,
}

impl FailureController {
    fn new(pattern: Vec<FailureEvent>) -> Self {
        Self {
            failure_sequence: pattern,
            current_request: AtomicUsize::new(0),
        }
    }

    fn next_response(&self) -> FailureEvent {
        let index = self.current_request.fetch_add(1, Ordering::SeqCst);
        self.failure_sequence[index % self.failure_sequence.len()].clone()
    }
}
```

### 11. Required Dependencies

The following dependencies should be added to `[dev-dependencies]` in `Cargo.toml`:

```toml
[dev-dependencies]
# Existing dependencies
tempfile = "3.8"
paste = "1.0"

# New dependencies for stress testing
httpmock = "0.7.0"              # HTTP mocking with deterministic failure injection
hyper = "0.14"                  # Small helper for TCP-level fault injection (~50 LOC)
tokio = { version = "1.0", features = ["rt", "net"], optional = true }  # Only for hyper helper
serde_json = "1.0"              # For test result serialization
fail = "0.5"                    # Failpoints for crash testing (test-only)
fd-lock = "4.0"                 # Cross-process lock validation
```

**Dependency Rationale**:
- **Minimal Impact**: Only 5 additional crates for comprehensive stress testing
- **Well-maintained**: All are industry standard with active development
- **Zero runtime impact**: Only in dev-dependencies, no production overhead
- **Compatible**: Work well with beaker's existing blocking/synchronous model
- **No timing dependencies**: Deterministic behavior without sleep or delays
- **Targeted async usage**: tokio only for the small hyper helper (~50 LOC), not for main test logic
- **Failpoint isolation**: `fail` crate used only in test scenarios, no production code changes

### 12. Test Configuration and Controls

**Environment Variables**:
```rust
// Control test execution behavior
BEAKER_STRESS_MAX_PROCESSES=20        // Max concurrent processes
BEAKER_STRESS_FAILURE_PATTERN=deterministic  // Use predefined failure sequences
BEAKER_STRESS_ENABLE_COREML=true      // Test CoreML on macOS
BEAKER_STRESS_CACHE_SIZE_LIMIT=1GB    // Cache size constraints
```

**Test Selection**:
```rust
// Granular test control for development
cargo test stress::concurrency::same_model
cargo test stress::failures::network_timeout
cargo test stress::coreml --features=macos-only
cargo test stress --release -- --test-threads=1  // For deterministic results
```

### 13. Success Criteria

**Functional Requirements**:
- ✅ No cache corruption under any concurrent access pattern with shared cache directories
- ✅ Lock files properly cleaned up after process termination in shared cache
- ✅ SHA-256 validation never passes for corrupted data
- ✅ Failed downloads don't leave partial files in shared cache
- ✅ Concurrent processes don't deadlock or hang indefinitely
- ✅ Exactly one final artifact exists per model in shared cache

**Logical Invariants**:
- ✅ Cache consistency across all successful processes (eventual property)
- ✅ Deterministic failure handling matches expected TCP/HTTP patterns
- ✅ All processes eventually complete (no infinite hanging)
- ✅ Error recovery follows expected paths for each failure type
- ✅ Resource cleanup completes for all process termination scenarios
- ✅ Readers never observe partial content (atomic write operations)
- ✅ Crash recovery properly handles write → fsync → rename boundaries

**Reliability Requirements**:
- ✅ Tests complete rapidly with immediate failure injection (under 30 seconds)
- ✅ No flaky test behavior - deterministic failure patterns
- ✅ Reproducible results for given failure injection sequences
- ✅ Clean process termination without resource leaks
- ✅ Comprehensive error reporting for debugging failures

### 14. Implementation Considerations

**Minimizing Development Impact**:
- Stress tests live entirely in `tests/stress/` directory
- No changes to production code paths except optional failpoint annotations (test-only)
- Dependencies on httpmock, hyper helper, and failpoints only in dev-dependencies
- Can be disabled for faster development builds
- Uses simple thread-based concurrency matching beaker's blocking model
- Small hyper helper (~50 LOC) isolated from main codebase

**Deterministic Testing Strategy**:
- Event-based coordination with barriers and channels instead of timing
- Predefined failure sequences ensure reproducible test outcomes
- Logical invariants validation rather than time-based assertions
- Immediate failure injection eliminates flaky timing dependencies
- Embedded fixtures eliminate network variance during test execution
- Shared cache directories enable real lock contention testing

**CI Integration**:
- Run basic concurrency tests on every PR (fast execution under 30 seconds)
- Extended deterministic stress patterns on nightly builds
- Platform-specific tests (CoreML on macOS runners)
- Logical invariant validation for reliable regression detection

**Maintenance and Evolution**:
- Framework designed to accommodate new models (issue #22)
- Extensible failure patterns for new failure modes
- Modular test scenarios for incremental development
- Documentation for adding new concurrency test cases

## Related Issues Integration

This stress testing framework is designed to work alongside and validate upcoming features:

### CLI Model Access Support
- **Integration Point**: Test framework validates concurrent access when users specify custom model URLs/paths via CLI options (--model-url, --model-path, --model-checksum)
- **Stress Scenarios**: Multiple processes using different models specified via CLI arguments
- **Validation**: Ensure cache isolation between different model URLs and local paths

### Cache Metadata Integration
- **Integration Point**: Leverage cache hit/miss metadata for stress test validation
- **Metrics Collection**: Use cache timing and hit rate data to validate performance under load
- **Instrumentation**: Cache metadata helps identify cache contention and performance bottlenecks

The framework integrates with these implemented features to provide comprehensive testing capabilities.

## Conclusion

This stress testing framework will provide comprehensive validation of beaker's caching mechanisms under realistic concurrent usage patterns. By using deterministic failure injection and event-based coordination, it eliminates timing dependencies while ensuring fast execution and reliable results.

The framework focuses on logical invariants and eventual properties that must hold, making tests both robust and maintainable. With immediate failure injection and barrier-based process coordination, tests complete rapidly without the flakiness inherent in time-based testing approaches.

The framework is designed to evolve alongside beaker's caching implementation, supporting future enhancements like CLI model selection (#22) and cache metadata improvements (#35), while maintaining deterministic test execution and comprehensive failure mode coverage for continuous integration.
