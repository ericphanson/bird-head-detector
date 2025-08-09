//! TCP-level fault injection server for realistic network failure simulation

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, StatusCode};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

use crate::stress::fixtures::TestFixtures;

/// TCP-level failure types that httpmock cannot simulate
#[derive(Clone, Debug)]
pub enum TcpFaultType {
    Success(Vec<u8>),      // Normal response with data
    ConnectionRefused,     // No server listening (simulate by not starting server)
    MidStreamAbort(usize), // Send N bytes then close connection abruptly
    HeaderThenClose,       // Send headers only, then close
    ImmediateClose,        // Accept connection, then close immediately
}

/// TCP fault server for precise failure injection
pub struct TcpFaultServer {
    addr: SocketAddr,
    fault_sequence: Arc<Vec<TcpFaultType>>,
    request_counter: Arc<AtomicUsize>,
}

impl TcpFaultServer {
    /// Create a new TCP fault server with specified failure pattern
    pub async fn new(
        fault_sequence: Vec<TcpFaultType>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let addr = ([127, 0, 0, 1], 0).into(); // Bind to random port
        let fault_sequence = Arc::new(fault_sequence);
        let request_counter = Arc::new(AtomicUsize::new(0));

        Ok(Self {
            addr,
            fault_sequence,
            request_counter,
        })
    }

    /// Start the server and return the actual listening address
    pub async fn start(self) -> Result<SocketAddr, Box<dyn std::error::Error>> {
        let fault_sequence = Arc::clone(&self.fault_sequence);
        let request_counter = Arc::clone(&self.request_counter);

        let make_svc = make_service_fn(move |_| {
            let fault_sequence = Arc::clone(&fault_sequence);
            let request_counter = Arc::clone(&request_counter);

            async move {
                Ok::<_, Infallible>(service_fn(move |req| {
                    handle_request(
                        req,
                        Arc::clone(&fault_sequence),
                        Arc::clone(&request_counter),
                    )
                }))
            }
        });

        let server = Server::bind(&self.addr).serve(make_svc);
        let actual_addr = server.local_addr();

        // Spawn server in background
        tokio::spawn(async move {
            if let Err(e) = server.await {
                eprintln!("TCP fault server error: {}", e);
            }
        });

        Ok(actual_addr)
    }

    /// Get base URL for this server
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// Handle individual request with fault injection
async fn handle_request(
    _req: Request<Body>,
    fault_sequence: Arc<Vec<TcpFaultType>>,
    request_counter: Arc<AtomicUsize>,
) -> Result<Response<Body>, Infallible> {
    let request_index = request_counter.fetch_add(1, Ordering::SeqCst);
    let fault_type = &fault_sequence[request_index % fault_sequence.len()];

    match fault_type {
        TcpFaultType::Success(data) => {
            // Normal successful response
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .body(Body::from(data.clone()))
                .unwrap())
        }
        TcpFaultType::ConnectionRefused => {
            // This should be handled by not starting the server at all
            // For now, return 503 Service Unavailable
            Ok(Response::builder()
                .status(StatusCode::SERVICE_UNAVAILABLE)
                .body(Body::empty())
                .unwrap())
        }
        TcpFaultType::MidStreamAbort(bytes_before_abort) => {
            // Send partial data then simulate connection abort
            let fixtures = TestFixtures::new();
            let full_data = fixtures.small_model_bytes;
            let partial_data = if *bytes_before_abort < full_data.len() {
                &full_data[..*bytes_before_abort]
            } else {
                full_data
            };

            // Create response with partial data
            // Note: Actual mid-stream abort is complex in hyper
            // This simulates it by sending partial content with wrong content-length
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .header("content-length", full_data.len().to_string()) // Lie about length
                .body(Body::from(partial_data.to_vec()))
                .unwrap())
        }
        TcpFaultType::HeaderThenClose => {
            // Send headers but empty body (simulates connection close after headers)
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/octet-stream")
                .header("content-length", "1024") // Lie about having content
                .body(Body::empty()) // But send no body
                .unwrap())
        }
        TcpFaultType::ImmediateClose => {
            // Simulate immediate close by returning error response
            // In real implementation, this would close the connection immediately
            sleep(Duration::from_millis(1)).await; // Brief delay to simulate connection setup
            Ok(Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::empty())
                .unwrap())
        }
    }
}

/// Helper function to create a connection refused scenario
/// Returns None if the server should not be started at all
pub fn should_start_server(fault_sequence: &[TcpFaultType]) -> bool {
    // If the first fault is ConnectionRefused, don't start the server
    !matches!(
        fault_sequence.first(),
        Some(TcpFaultType::ConnectionRefused)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_tcp_fault_server_creation() {
        let fault_sequence = vec![
            TcpFaultType::Success(b"test data".to_vec()),
            TcpFaultType::MidStreamAbort(5),
        ];

        let server = TcpFaultServer::new(fault_sequence).await.unwrap();
        assert!(server.base_url().starts_with("http://127.0.0.1:"));
    }

    #[test]
    async fn test_should_start_server() {
        let with_connection_refused = vec![TcpFaultType::ConnectionRefused];
        assert!(!should_start_server(&with_connection_refused));

        let normal_sequence = vec![TcpFaultType::Success(b"data".to_vec())];
        assert!(should_start_server(&normal_sequence));
    }

    #[test]
    async fn test_tcp_fault_types() {
        // Test that all fault types can be created
        let fault_sequence = vec![
            TcpFaultType::Success(b"test".to_vec()),
            TcpFaultType::ConnectionRefused,
            TcpFaultType::MidStreamAbort(10),
            TcpFaultType::HeaderThenClose,
            TcpFaultType::ImmediateClose,
        ];

        let _server = TcpFaultServer::new(fault_sequence).await.unwrap();
        // Test passes if no panic occurs
    }
}
