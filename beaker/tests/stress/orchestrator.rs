//! Process orchestration for concurrent stress testing

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Barrier};
use std::thread;
use tempfile::TempDir;

use crate::stress::mock_servers::{FailureEvent, MockServerManager};

/// Result from executing a beaker process
#[derive(Debug)]
pub struct ProcessResult {
    pub process_id: usize,
    pub exit_code: i32,
    pub shared_cache_dir: PathBuf,
    pub output_dir: PathBuf,
    pub error_output: String,
    pub stdout_output: String,
}

/// Cache strategy for test scenarios
#[derive(Clone)]
pub enum CacheStrategy {
    Shared(PathBuf),          // All processes use same cache dir (for contention testing)
    PerProcess(Vec<PathBuf>), // Each process gets own cache dir (for isolation testing)
}

/// Test scenario types
#[derive(Clone, Debug)]
pub enum TestScenario {
    SameModelContention,
    DifferentModelContention,
    NetworkFailureRecovery,
    CorruptionHandling,
    CacheDirectoryStress,
    DifferentModelIsolation,
    CoreMLConcurrency,
}

/// Stress test orchestrator for managing concurrent beaker processes
pub struct StressTestOrchestrator {
    max_concurrent_processes: usize,
    temp_base: TempDir,
}

impl StressTestOrchestrator {
    pub fn new(max_concurrent_processes: usize) -> std::io::Result<Self> {
        Ok(Self {
            max_concurrent_processes,
            temp_base: TempDir::new()?,
        })
    }

    pub fn run_stress_test(
        &self,
        scenario: TestScenario,
        failure_patterns: Vec<Vec<FailureEvent>>,
    ) -> StressTestResults {
        let cache_strategy = self.configure_cache_strategy(&scenario);

        // Set up mock server with combined failure pattern
        let combined_pattern: Vec<FailureEvent> =
            failure_patterns.iter().flatten().cloned().collect();
        let mock_manager = MockServerManager::new(combined_pattern);

        // Synchronization for simultaneous start
        let start_barrier = Arc::new(Barrier::new(self.max_concurrent_processes + 1));
        let (result_tx, result_rx): (Sender<ProcessResult>, Receiver<ProcessResult>) = channel();

        // Launch concurrent beaker processes
        let handles: Vec<_> = (0..self.max_concurrent_processes)
            .map(|i| {
                let barrier = Arc::clone(&start_barrier);
                let cache_strategy = cache_strategy.clone();
                let mock_url = format!("{}/small-test-model.onnx", mock_manager.http_base_url());
                let tx = result_tx.clone();
                let temp_base_path = self.temp_base.path().to_path_buf();
                let failure_pattern = failure_patterns
                    .get(i % failure_patterns.len())
                    .cloned()
                    .unwrap_or_default();

                thread::spawn(move || {
                    // Wait for all processes to be ready
                    barrier.wait();

                    // Execute beaker process
                    let result = execute_beaker_process(
                        i,
                        &cache_strategy,
                        &mock_url,
                        &temp_base_path,
                        &failure_pattern,
                    );

                    tx.send(result).unwrap();
                })
            })
            .collect();

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

    fn configure_cache_strategy(&self, scenario: &TestScenario) -> CacheStrategy {
        match scenario {
            TestScenario::SameModelContention
            | TestScenario::NetworkFailureRecovery
            | TestScenario::CorruptionHandling
            | TestScenario::CacheDirectoryStress
            | TestScenario::CoreMLConcurrency => {
                let shared_cache = self.temp_base.path().join("shared_cache");
                std::fs::create_dir_all(&shared_cache).unwrap();
                CacheStrategy::Shared(shared_cache)
            }
            TestScenario::DifferentModelContention | TestScenario::DifferentModelIsolation => {
                let per_process_dirs: Vec<PathBuf> = (0..self.max_concurrent_processes)
                    .map(|i| {
                        let dir = self.temp_base.path().join(format!("cache_{}", i));
                        std::fs::create_dir_all(&dir).unwrap();
                        dir
                    })
                    .collect();
                CacheStrategy::PerProcess(per_process_dirs)
            }
        }
    }
}

/// Execute a single beaker process with specified configuration
fn execute_beaker_process(
    process_id: usize,
    cache_strategy: &CacheStrategy,
    mock_url: &str,
    temp_base: &PathBuf,
    _failure_pattern: &[FailureEvent],
) -> ProcessResult {
    let output_dir = temp_base.join(format!("output_{}", process_id));
    std::fs::create_dir_all(&output_dir).unwrap();

    let cache_dir = match cache_strategy {
        CacheStrategy::Shared(dir) => dir.clone(),
        CacheStrategy::PerProcess(dirs) => dirs[process_id].clone(),
    };

    // Create a test image for processing
    let test_image = create_test_image(&output_dir);

    // Execute beaker cutout command with model URL parameter
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--",
            "cutout",
            test_image.to_str().unwrap(),
            "--metadata",
            "--model-url",
            mock_url,
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .current_dir(temp_base.join("../../../beaker")) // Navigate to beaker directory
        .env("ONNX_MODEL_CACHE_DIR", cache_dir.to_str().unwrap())
        .output()
        .unwrap_or_else(|e| Output {
            status: std::process::ExitStatus::default(),
            stdout: Vec::new(),
            stderr: format!("Failed to execute: {}", e).into_bytes(),
        });

    let exit_code = output.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();

    ProcessResult {
        process_id,
        exit_code,
        shared_cache_dir: cache_dir,
        output_dir,
        error_output: stderr,
        stdout_output: stdout,
    }
}

/// Create a minimal test image file
fn create_test_image(output_dir: &PathBuf) -> PathBuf {
    let test_image = output_dir.join("test_image.jpg");

    // Create a minimal JPEG-like file (just for testing, doesn't need to be valid)
    let jpeg_header = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]; // JPEG SOI + App0 marker start
    std::fs::write(&test_image, jpeg_header).unwrap();

    test_image
}

/// Results from stress testing
pub struct StressTestResults {
    pub results: Vec<ProcessResult>,
    pub success_count: usize,
    pub failure_count: usize,
}

impl StressTestResults {
    pub fn new(results: Vec<ProcessResult>) -> Self {
        let success_count = results.iter().filter(|r| r.exit_code == 0).count();
        let failure_count = results.len() - success_count;

        Self {
            results,
            success_count,
            failure_count,
        }
    }

    pub fn total_processes(&self) -> usize {
        self.results.len()
    }

    pub fn get_successful_results(&self) -> Vec<&ProcessResult> {
        self.results.iter().filter(|r| r.exit_code == 0).collect()
    }

    pub fn get_failed_results(&self) -> Vec<&ProcessResult> {
        self.results.iter().filter(|r| r.exit_code != 0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orchestrator_creation() {
        let orchestrator = StressTestOrchestrator::new(5).unwrap();
        assert_eq!(orchestrator.max_concurrent_processes, 5);
    }

    #[test]
    fn test_cache_strategy_configuration() {
        let orchestrator = StressTestOrchestrator::new(3).unwrap();

        // Shared cache scenarios
        let shared_scenario = TestScenario::SameModelContention;
        if let CacheStrategy::Shared(_) = orchestrator.configure_cache_strategy(&shared_scenario) {
            // Expected
        } else {
            panic!("Expected shared cache strategy");
        }

        // Per-process cache scenarios
        let isolated_scenario = TestScenario::DifferentModelIsolation;
        if let CacheStrategy::PerProcess(dirs) =
            orchestrator.configure_cache_strategy(&isolated_scenario)
        {
            assert_eq!(dirs.len(), 3);
        } else {
            panic!("Expected per-process cache strategy");
        }
    }

    #[test]
    fn test_stress_test_results() {
        let results = vec![
            ProcessResult {
                process_id: 0,
                exit_code: 0,
                shared_cache_dir: PathBuf::from("/tmp/cache"),
                output_dir: PathBuf::from("/tmp/out0"),
                error_output: String::new(),
                stdout_output: String::new(),
            },
            ProcessResult {
                process_id: 1,
                exit_code: 1,
                shared_cache_dir: PathBuf::from("/tmp/cache"),
                output_dir: PathBuf::from("/tmp/out1"),
                error_output: "Error".to_string(),
                stdout_output: String::new(),
            },
        ];

        let stress_results = StressTestResults::new(results);
        assert_eq!(stress_results.success_count, 1);
        assert_eq!(stress_results.failure_count, 1);
        assert_eq!(stress_results.total_processes(), 2);
    }
}
