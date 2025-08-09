//! Parallel process stress testing framework for beaker's cache mechanisms
//!
//! This module provides comprehensive testing of ONNX and CoreML caches under
//! concurrent access with simulated network failures and crash recovery.

pub mod fixtures;
pub mod mock_servers;
pub mod orchestrator;
pub mod performance;
pub mod tcp_fault_server;
pub mod validators;

// Re-export main types for convenience (when needed)
pub use fixtures::TestFixtures;
pub use mock_servers::{FailureEvent, MockServerManager};
pub use orchestrator::{ProcessResult, StressTestOrchestrator, TestScenario};
pub use performance::{PerformanceBenchmark, PerformanceMetrics, PerformanceTracker};
pub use tcp_fault_server::{TcpFaultServer, TcpFaultType};
pub use validators::CacheValidator;
