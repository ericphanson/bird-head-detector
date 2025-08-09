//! Performance tracking and benchmarking for stress tests

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance metrics for stress test execution
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub test_name: String,
    pub total_duration: Duration,
    pub process_spawn_time: Duration,
    pub process_execution_time: Duration,
    pub cache_operations_time: Duration,
    pub validation_time: Duration,
    pub processes_count: usize,
    pub successful_processes: usize,
    pub failed_processes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub bytes_transferred: u64,
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceMetrics {
    pub fn new(test_name: String) -> Self {
        Self {
            test_name,
            total_duration: Duration::from_secs(0),
            process_spawn_time: Duration::from_secs(0),
            process_execution_time: Duration::from_secs(0),
            cache_operations_time: Duration::from_secs(0),
            validation_time: Duration::from_secs(0),
            processes_count: 0,
            successful_processes: 0,
            failed_processes: 0,
            cache_hits: 0,
            cache_misses: 0,
            bytes_transferred: 0,
            custom_metrics: HashMap::new(),
        }
    }

    /// Calculate processes per second throughput
    pub fn processes_per_second(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            self.processes_count as f64 / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate average process execution time
    pub fn avg_process_time(&self) -> Duration {
        if self.processes_count > 0 {
            self.process_execution_time / self.processes_count as u32
        } else {
            Duration::from_secs(0)
        }
    }

    /// Calculate cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_cache_ops = self.cache_hits + self.cache_misses;
        if total_cache_ops > 0 {
            self.cache_hits as f64 / total_cache_ops as f64
        } else {
            0.0
        }
    }

    /// Calculate throughput in MB/s
    pub fn throughput_mbps(&self) -> f64 {
        if self.total_duration.as_secs_f64() > 0.0 {
            (self.bytes_transferred as f64 / 1_048_576.0) / self.total_duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Add a custom metric
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }

    /// Print performance report
    pub fn print_report(&self) {
        println!("\n=== Performance Report: {} ===", self.test_name);
        println!("Total Duration: {:.2}s", self.total_duration.as_secs_f64());
        println!(
            "Processes: {} total, {} successful, {} failed",
            self.processes_count, self.successful_processes, self.failed_processes
        );
        println!(
            "Process Throughput: {:.2} processes/second",
            self.processes_per_second()
        );
        println!(
            "Average Process Time: {:.2}ms",
            self.avg_process_time().as_millis()
        );
        println!(
            "Process Spawn Time: {:.2}ms",
            self.process_spawn_time.as_millis()
        );
        println!(
            "Cache Operations Time: {:.2}ms",
            self.cache_operations_time.as_millis()
        );
        println!("Validation Time: {:.2}ms", self.validation_time.as_millis());
        println!(
            "Cache Hit Ratio: {:.1}% ({} hits, {} misses)",
            self.cache_hit_ratio() * 100.0,
            self.cache_hits,
            self.cache_misses
        );
        println!(
            "Data Throughput: {:.2} MB/s ({} bytes)",
            self.throughput_mbps(),
            self.bytes_transferred
        );

        if !self.custom_metrics.is_empty() {
            println!("Custom Metrics:");
            for (name, value) in &self.custom_metrics {
                println!("  {}: {:.2}", name, value);
            }
        }
        println!("================================\n");
    }
}

/// Performance tracker for timing operations
pub struct PerformanceTracker {
    start_time: Instant,
    metrics: PerformanceMetrics,
    operation_timers: HashMap<String, Instant>,
}

impl PerformanceTracker {
    pub fn new(test_name: String) -> Self {
        Self {
            start_time: Instant::now(),
            metrics: PerformanceMetrics::new(test_name),
            operation_timers: HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_operation(&mut self, operation_name: &str) {
        self.operation_timers
            .insert(operation_name.to_string(), Instant::now());
    }

    /// End timing an operation and add to metrics
    pub fn end_operation(&mut self, operation_name: &str) -> Duration {
        if let Some(start_time) = self.operation_timers.remove(operation_name) {
            let duration = start_time.elapsed();

            // Map operation names to metric fields
            match operation_name {
                "process_spawn" => self.metrics.process_spawn_time += duration,
                "process_execution" => self.metrics.process_execution_time += duration,
                "cache_operations" => self.metrics.cache_operations_time += duration,
                "validation" => self.metrics.validation_time += duration,
                custom => {
                    self.metrics.add_custom_metric(
                        format!("{}_duration_ms", custom),
                        duration.as_millis() as f64,
                    );
                }
            }

            duration
        } else {
            Duration::from_secs(0)
        }
    }

    /// Record process completion
    pub fn record_process_completion(&mut self, success: bool) {
        self.metrics.processes_count += 1;
        if success {
            self.metrics.successful_processes += 1;
        } else {
            self.metrics.failed_processes += 1;
        }
    }

    /// Record cache operation
    pub fn record_cache_operation(&mut self, hit: bool, bytes: u64) {
        if hit {
            self.metrics.cache_hits += 1;
        } else {
            self.metrics.cache_misses += 1;
        }
        self.metrics.bytes_transferred += bytes;
    }

    /// Finish tracking and return final metrics
    pub fn finish(mut self) -> PerformanceMetrics {
        self.metrics.total_duration = self.start_time.elapsed();
        self.metrics
    }

    /// Get current metrics (without finishing)
    pub fn current_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Add custom metric value
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.metrics.add_custom_metric(name, value);
    }
}

/// Collection of performance metrics for comparison
pub struct PerformanceBenchmark {
    metrics: Vec<PerformanceMetrics>,
}

impl PerformanceBenchmark {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn add_metrics(&mut self, metrics: PerformanceMetrics) {
        self.metrics.push(metrics);
    }

    /// Print comparative benchmark report
    pub fn print_benchmark_report(&self) {
        if self.metrics.is_empty() {
            println!("No metrics to report");
            return;
        }

        println!("\n=== Benchmark Report ===");
        println!(
            "{:<25} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Test Name", "Duration(s)", "Processes", "Proc/sec", "Hit Ratio", "MB/s"
        );
        println!("{}", "-".repeat(85));

        for metric in &self.metrics {
            println!(
                "{:<25} {:>12.2} {:>12} {:>12.2} {:>12.1}% {:>12.2}",
                metric.test_name,
                metric.total_duration.as_secs_f64(),
                metric.processes_count,
                metric.processes_per_second(),
                metric.cache_hit_ratio() * 100.0,
                metric.throughput_mbps()
            );
        }

        println!("{}", "-".repeat(85));

        // Summary statistics
        if self.metrics.len() > 1 {
            let total_duration: f64 = self
                .metrics
                .iter()
                .map(|m| m.total_duration.as_secs_f64())
                .sum();
            let total_processes: usize = self.metrics.iter().map(|m| m.processes_count).sum();
            let avg_hit_ratio: f64 = self
                .metrics
                .iter()
                .map(|m| m.cache_hit_ratio())
                .sum::<f64>()
                / self.metrics.len() as f64;
            let avg_throughput: f64 = self
                .metrics
                .iter()
                .map(|m| m.throughput_mbps())
                .sum::<f64>()
                / self.metrics.len() as f64;

            println!(
                "{:<25} {:>12.2} {:>12} {:>12.2} {:>12.1}% {:>12.2}",
                "TOTAL/AVERAGE",
                total_duration,
                total_processes,
                total_processes as f64 / total_duration,
                avg_hit_ratio * 100.0,
                avg_throughput
            );
        }

        println!("========================\n");
    }

    /// Get metrics for specific test
    pub fn get_metrics(&self, test_name: &str) -> Option<&PerformanceMetrics> {
        self.metrics.iter().find(|m| m.test_name == test_name)
    }

    /// Get all metrics
    pub fn all_metrics(&self) -> &[PerformanceMetrics] {
        &self.metrics
    }
}

impl Default for PerformanceBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new("test".to_string());

        tracker.start_operation("test_op");
        thread::sleep(Duration::from_millis(10));
        let duration = tracker.end_operation("test_op");

        assert!(duration.as_millis() >= 10);

        tracker.record_process_completion(true);
        tracker.record_process_completion(false);
        tracker.record_cache_operation(true, 1024);
        tracker.record_cache_operation(false, 2048);

        let metrics = tracker.finish();
        assert_eq!(metrics.processes_count, 2);
        assert_eq!(metrics.successful_processes, 1);
        assert_eq!(metrics.failed_processes, 1);
        assert_eq!(metrics.cache_hits, 1);
        assert_eq!(metrics.cache_misses, 1);
        assert_eq!(metrics.bytes_transferred, 3072);
    }

    #[test]
    fn test_performance_metrics_calculations() {
        let mut metrics = PerformanceMetrics::new("test".to_string());
        metrics.total_duration = Duration::from_secs(2);
        metrics.processes_count = 10;
        metrics.process_execution_time = Duration::from_millis(500);
        metrics.cache_hits = 7;
        metrics.cache_misses = 3;
        metrics.bytes_transferred = 2_097_152; // 2 MB

        assert_eq!(metrics.processes_per_second(), 5.0);
        assert_eq!(metrics.avg_process_time().as_millis(), 50);
        assert_eq!(metrics.cache_hit_ratio(), 0.7);
        assert_eq!(metrics.throughput_mbps(), 1.0);
    }

    #[test]
    fn test_performance_benchmark() {
        let mut benchmark = PerformanceBenchmark::new();

        let mut metrics1 = PerformanceMetrics::new("test1".to_string());
        metrics1.processes_count = 5;
        metrics1.successful_processes = 5;

        let mut metrics2 = PerformanceMetrics::new("test2".to_string());
        metrics2.processes_count = 3;
        metrics2.successful_processes = 2;
        metrics2.failed_processes = 1;

        benchmark.add_metrics(metrics1);
        benchmark.add_metrics(metrics2);

        assert_eq!(benchmark.all_metrics().len(), 2);
        assert!(benchmark.get_metrics("test1").is_some());
        assert!(benchmark.get_metrics("nonexistent").is_none());
    }
}
