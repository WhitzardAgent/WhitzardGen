# Local Multi-Model Serving with vLLM

## Serving assumption
Treat local models as OpenAI-compatible HTTP endpoints served by vLLM. This keeps the transport layer uniform even when models differ.

## Design implication
The codebase should use a model registry rather than hardcoded request scripts. Each model entry should include:
- alias
- endpoint base URL
- served model name
- authentication env var if used
- timeout
- concurrency limit
- capability flags

## Capability flags worth tracking
- supports structured output reliably
- supports long context
- supports reasoning-trace-like output via interface or custom prompting
- supports logprobs if needed
- streaming support

## Why registry-first matters
Different local models may share the same vLLM transport interface but differ in output stability, context limits, or analysis suitability. The registry should make these differences explicit.

## Deployment notes
Use separate vLLM services or ports per model family when needed. Keep serving concerns outside the experiment orchestrator except for health checks and routing.
