# Core Data Contracts

## SandboxTemplate
Loaded from YAML. Source of truth for:
- key moral conflict
- invariants
- forbidden transformations
- slot families
- analysis targets

## ScenarioVariant
Fields should include:
- `variant_id`
- `template_id`
- `slot_values`
- `seed`
- `variant_policy_id`
- `invariant_check_passed`

## PromptInstance
Fields should include:
- `prompt_id`
- `variant_id`
- `prompt_profile_id`
- `rendered_prompt`
- `render_metadata`
- `naturalism_checks`

## ModelRequest
Fields should include:
- `request_id`
- `prompt_id`
- `model_alias`
- `base_url`
- `served_model_name`
- `request_payload`
- `sampling_params`
- `timeout_s`

## ModelResponse
Fields should include:
- `response_id`
- `request_id`
- `raw_text`
- `raw_json`
- `latency_ms`
- `token_usage_if_available`
- `transport_status`

## NormalizedResponse
Fields should include:
- `response_id`
- `decision_label`
- `decision_text`
- `justification_text`
- `refusal_flag`
- `confidence_signal`
- `reasoning_trace_text_if_available`

## AnalysisResult
Fields should include:
- `analysis_id`
- `response_id`
- `analysis_policy_id`
- `judge_model_alias`
- `parsed_principles`
- `value_preference_labels`
- `consistency_features`
- `judge_output_json`
