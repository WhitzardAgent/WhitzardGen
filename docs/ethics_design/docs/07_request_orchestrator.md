# Request Orchestrator

## Responsibilities
- batch prompt instances by model
- send concurrent requests with per-model limits
- record raw request and response artifacts
- support retries and resumability
- avoid duplicate execution when rerunning a partially completed run

## Key design requirements
- idempotent run manifests
- request hashing for deduplication
- explicit retry policy per transport failure class
- per-model backpressure and concurrency control
- optional streaming support without changing higher-level contracts

## Recommended implementation pattern
- `RunPlanner` builds the execution graph
- `RequestExecutor` handles async transport
- `ArtifactWriter` persists raw and normalized artifacts
- `ResumeScanner` detects missing or failed units and reconstructs remaining work
