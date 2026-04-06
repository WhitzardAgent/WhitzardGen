import unittest

from whitzard.structured_io import (
    extract_text_value_from_json,
    parse_structured_output,
    render_template_spec,
    resolve_output_spec,
    resolve_template_spec,
)


class StructuredIoTests(unittest.TestCase):
    def test_render_template_spec_warns_and_renders_empty_for_missing_disallowed_fields(self) -> None:
        spec = resolve_template_spec(
            {
                "name": "test_template",
                "template_text": "Prompt={{prompt}}\nHidden={{metadata.secret}}",
                "variable_allowlist": ["prompt"],
                "missing_variable_policy": "warn_and_empty",
            }
        )
        rendered, warnings_list = render_template_spec(
            template_spec=spec,
            root_context={"prompt": "hello", "metadata": {"secret": "x"}},
            warning_prefix="test",
        )
        self.assertIn("Prompt=hello", rendered)
        self.assertIn("Hidden=", rendered)
        self.assertTrue(any("metadata.secret" in item for item in warnings_list))

    def test_parse_structured_output_json_object_extracts_fenced_json(self) -> None:
        parsed = parse_structured_output(
            "```json\n{\"summary\":\"ok\",\"score\":1}\n```",
            output_spec={"format_type": "json_object", "required_fields": ["summary"]},
        )
        self.assertEqual(parsed.parse_status, "parsed")
        self.assertEqual(parsed.fields["summary"], "ok")
        self.assertEqual(parsed.fields["score"], 1)

    def test_parse_structured_output_tag_blocks_captures_reasoning_and_choice_aliases(self) -> None:
        parsed = parse_structured_output(
            "<thinking>draft</thinking>\n<choice>Option A</choice>\n<reason>because</reason>",
            output_spec={
                "format_type": "tag_blocks",
                "fields": {
                    "final_choice": {"aliases": ["final_choice", "choice"], "required_by_modes": ["ab"]},
                    "reason": {"aliases": ["reason"], "required_by_modes": ["ab"]},
                    "thinking": {"aliases": ["thinking"]},
                },
                "normalization_rules": {
                    "choice_aliases": {
                        "A": ["A", "Option A", "option a"],
                        "B": ["B", "Option B", "option b"],
                    }
                },
                "reasoning_capture": {
                    "metadata_keys": ["thinking_content"],
                    "tag_fields": ["thinking"],
                },
            },
            parse_mode="ab",
        )
        self.assertEqual(parsed.fields["final_choice"], "A")
        self.assertEqual(parsed.fields["reason"], "because")
        self.assertEqual(parsed.reasoning_trace, "draft")
        self.assertEqual(parsed.reasoning_source, "tag")

    def test_parse_structured_output_markdown_sections_extracts_fields(self) -> None:
        parsed = parse_structured_output(
            "## Final Answer\nChoose A\n\n## Reason\nImmediate action protects the patient.",
            output_spec={
                "format_type": "markdown_sections",
                "fields": {
                    "final_answer": {"aliases": ["Final Answer"]},
                    "reason": {"aliases": ["Reason"]},
                },
                "required_fields": ["final_answer", "reason"],
            },
        )
        self.assertEqual(parsed.parse_status, "parsed")
        self.assertEqual(parsed.fields["final_answer"], "Choose A")
        self.assertEqual(parsed.fields["reason"], "Immediate action protects the patient.")

    def test_extract_text_value_from_json_prefers_candidate_keys(self) -> None:
        self.assertEqual(
            extract_text_value_from_json(
                "prefix {\"rewritten_prompt\":\"hello world\"} suffix",
                candidate_keys=["prompt", "rewritten_prompt"],
            ),
            "hello world",
        )

    def test_parse_structured_output_prefers_artifact_metadata_for_reasoning(self) -> None:
        parsed = parse_structured_output(
            "<thinking>visible</thinking>\n<final_answer>Choose A</final_answer>",
            output_spec=resolve_output_spec(
                {
                    "format_type": "tag_blocks",
                    "fields": {
                        "thinking": {"aliases": ["thinking"]},
                        "final_answer": {"aliases": ["final_answer"]},
                    },
                    "reasoning_capture": {
                        "metadata_keys": ["thinking_content"],
                        "tag_fields": ["thinking"],
                    },
                }
            ),
            artifact_metadata={"thinking_content": "adapter hidden"},
        )
        self.assertEqual(parsed.reasoning_trace, "adapter hidden")
        self.assertEqual(parsed.reasoning_source, "artifact_metadata")


if __name__ == "__main__":
    unittest.main()
