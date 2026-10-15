"""Offline Bedrock Converse adapter checks; the client is always mocked."""

import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

from backend.app.bedrock import BedrockProvider
from backend.app.llm import LLMMessage, LLMRequest, ProviderFailure, StructuredLLM
from backend.app.agent import AgentDecision


FINAL = '{"decision":{"kind":"final"}}'


def completion(text=FINAL, reason="end_turn", usage=None):
    result = {"stopReason": reason, "output": {"message": {
        "role": "assistant", "content": [{"text": text}]}}}
    if usage is not None:
        result["usage"] = usage
    return result


class BedrockTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.converse.return_value = completion()
        self.provider = BedrockProvider(model_id="test.model-v1", client=self.client)
        self.request = LLMRequest(messages=(
            LLMMessage(role="system", content="Business instructions"),
            LLMMessage(role="user", content="Question"),
            LLMMessage(role="assistant", content=FINAL),
            LLMMessage(role="user", content="Follow-up")), max_output_tokens=2048)

    def result(self):
        return StructuredLLM(self.provider).generate(self.request, AgentDecision)

    def test_converse_messages_schema_budget_and_no_native_tools(self):
        schema = AgentDecision.model_json_schema()
        before = json.dumps(schema, sort_keys=True)
        response = self.provider.generate(self.request, response_schema=schema)
        self.assertEqual(response.text, FINAL)
        self.client.converse.assert_called_once()
        body = self.client.converse.call_args.kwargs
        self.assertEqual(body["modelId"], "test.model-v1")
        self.assertEqual(body["inferenceConfig"], {"maxTokens": 2048})
        self.assertEqual(body["messages"], [
            {"role": "user", "content": [{"text": "Question"}]},
            {"role": "assistant", "content": [{"text": FINAL}]},
            {"role": "user", "content": [{"text": "Follow-up"}]}])
        self.assertEqual(body["system"][0], {"text": "Business instructions"})
        self.assertEqual(json.loads(body["system"][-1]["text"].split("JSON Schema: ", 1)[1]), schema)
        self.assertEqual(json.dumps(schema, sort_keys=True), before)
        self.assertNotIn("toolConfig", body)

    def test_stop_reasons_usage_and_reasoning(self):
        self.client.converse.return_value = completion(usage={
            "inputTokens": 4, "outputTokens": 5, "totalTokens": 9})
        self.assertTrue(self.result().ok)
        self.assertEqual(self.provider.generate(self.request, response_schema={}).usage.model_dump(),
                         {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9})
        self.client.converse.return_value = completion(usage={
            "inputTokens": True, "outputTokens": -1, "totalTokens": "unknown"})
        self.assertEqual(self.provider.generate(self.request, response_schema={}).usage.model_dump(),
                         {"input_tokens": None, "output_tokens": None, "total_tokens": None})
        for reason, finish in (("end_turn", "stop"), ("stop_sequence", "stop"),
                               ("max_tokens", "length"),
                               ("model_context_window_exceeded", "length"),
                               ("content_filtered", "refusal"),
                               ("guardrail_intervened", "refusal")):
            with self.subTest(reason=reason):
                self.client.converse.return_value = completion(reason=reason)
                self.assertEqual(self.provider.generate(self.request, response_schema={}).finish_reason,
                                 finish)
        self.client.converse.return_value = completion()
        self.client.converse.return_value["output"]["message"]["content"].insert(
            0, {"reasoningContent": {"reasoningText": {"text": "PRIVATE"}}})
        self.assertEqual(self.provider.generate(self.request, response_schema={}).text, FINAL)

    def test_unknown_stop_tool_use_and_malformed_output_fail_closed(self):
        for payload in ({}, [], completion(reason="tool_use"),
                        {"stopReason": "end_turn", "output": {"message": {
                            "role": "assistant", "content": [{"toolUse": {"name": "approve"}}]}}},
                        completion(reason="malformed_model_output")):
            with self.subTest(payload=payload):
                self.client.converse.return_value = payload
                self.assertEqual(self.result().code, "INVALID_RESPONSE")

    def test_safe_errors_without_retry(self):
        class ServiceError(Exception):
            def __init__(self, code):
                super().__init__("secret provider detail")
                self.response = {"Error": {"Code": code}}

        for error, code in ((TimeoutError("secret"), "TIMEOUT"),
                            (ServiceError("ModelTimeoutException"), "TIMEOUT"),
                            (ServiceError("AccessDeniedException"), "UNAVAILABLE")):
            with self.subTest(code=code):
                self.client.converse.reset_mock(side_effect=True)
                self.client.converse.side_effect = error
                with self.assertRaises(ProviderFailure) as caught:
                    self.provider.generate(self.request, response_schema={})
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn("secret", str(caught.exception))
                self.client.converse.assert_called_once()
        self.client.converse.side_effect = RuntimeError("adapter bug")
        with self.assertRaisesRegex(RuntimeError, "adapter bug"):
            self.provider.generate(self.request, response_schema={})

    def test_invalid_input_before_client_call(self):
        for kwargs in ({"model_id": " "}, {"model_id": "test", "timeout_ms": 0},
                       {"model_id": "test", "timeout_ms": True},
                       {"model_id": "test", "timeout_ms": 120001}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BedrockProvider(client=self.client, **kwargs)
        with self.assertRaises(ValueError):
            self.provider.generate(LLMRequest(messages=(LLMMessage(role="system", content="only"),)),
                                   response_schema={})
        with self.assertRaises(ValueError):
            self.provider.generate(LLMRequest(messages=(
                LLMMessage(role="user", content="first"),
                LLMMessage(role="system", content="late"))), response_schema={})
        self.client.converse.assert_not_called()

    def test_owned_client_uses_bounded_timeouts_and_one_attempt(self):
        constructor = Mock()
        config_constructor = Mock(return_value=object())
        boto3 = types.ModuleType("boto3")
        boto3.client = constructor
        botocore = types.ModuleType("botocore")
        botocore.__path__ = []
        config = types.ModuleType("botocore.config")
        config.Config = config_constructor
        with patch.dict(sys.modules, {"boto3": boto3, "botocore": botocore,
                                      "botocore.config": config}):
            with BedrockProvider(model_id="test.model-v1", region_name="us-east-1",
                                 timeout_ms=1500):
                pass
        config_constructor.assert_called_once_with(
            connect_timeout=1.5, read_timeout=1.5,
            retries={"total_max_attempts": 1})
        constructor.assert_called_once_with("bedrock-runtime", region_name="us-east-1",
                                            config=config_constructor.return_value)
        constructor.return_value.close.assert_called_once()
        self.provider.close()
        self.client.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
