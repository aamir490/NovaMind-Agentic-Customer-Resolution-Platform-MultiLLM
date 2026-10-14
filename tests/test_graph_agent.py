"""Phase 9 graph routing and parity tests; all generation is scripted and local."""

import os
from unittest.mock import patch

from langgraph.errors import GraphRecursionError
from pydantic import ValidationError

import test_agent as baseline
from backend.app.agent import AgentConfig, ResolutionAgent
from backend.app.graph_agent import GraphResolutionAgent, GraphState
from backend.app.llm import StructuredLLM


class GraphAgentTests(baseline.AgentTests):
    # Reuse all 15 Phase 8 scenarios against the graph, not the old loop.
    def run_script(self, script, max_steps=8):
        self.provider = baseline.ScriptedProvider(script)
        self.agent = GraphResolutionAgent(StructuredLLM(self.provider), self.tools, AgentConfig(max_steps=max_steps))
        self.state = self.agent.run_state(self.request)
        return self.state.final_result

    def test_explicit_routing_and_loop_nodes(self):
        self.run_script([baseline.call("get_inventory", sku="LAP-1"), baseline.FINAL])
        self.assertEqual(self.state.node_path, (
            "load_case", "reason", "execute_tool", "record_result", "reason", "finalize",
        ))
        self.assertEqual(GraphState.model_validate_json(self.state.model_dump_json()), self.state)
        self.assertIsNone(self.state.staged_event)
        self.assertIsNone(self.agent._graph.checkpointer)
        self.assertIsNone(self.agent._graph.store)

    def test_failure_edges_keep_audit_once(self):
        self.run_script([baseline.call("approve")])
        self.assertEqual(self.state.node_path, ("load_case", "reason", "execute_tool", "record_result", "fail_safely"))
        self.assertEqual(len(self.state.history), 2)
        self.assertEqual(self.state.history[-1].error, "UNKNOWN_TOOL")
        self.run_script(["malformed"])
        self.assertEqual(self.state.node_path, ("load_case", "reason", "fail_safely"))
        self.assertEqual(len(self.state.history), 2)

    def test_final_on_last_step_succeeds_without_extra_reasoning(self):
        result = self.run_script([baseline.call("get_inventory", sku="LAP-1"), baseline.FINAL], max_steps=2)
        self.assertEqual(result.status, "INFORMATIONAL")
        self.assertEqual(len(self.provider.requests), 2)

    def test_long_run_uses_application_limit_not_default_graph_limit(self):
        result = self.run_script([baseline.call("get_inventory", sku="LAP-1")] * 32, max_steps=32)
        self.assertEqual(result.error, "STEP_LIMIT")
        self.assertEqual(len(self.provider.requests), 32)
        self.assertEqual(len(self.state.history), 33)
        self.assertEqual(self.state.node_path[-1], "fail_safely")

    def test_behavioral_parity_for_read_results_and_failures(self):
        scripts = [
            [baseline.FINAL],
            [baseline.call("get_inventory", sku="LAP-1"), baseline.FINAL],
            [baseline.call("approve")],
            ["bad JSON"],
            [baseline.call("get_order", order_id="bad")],
            [baseline.call("get_inventory", sku="LAP-1")] * 2,
        ]
        for script in scripts:
            with self.subTest(script=script):
                original_provider = baseline.ScriptedProvider(script)
                original = ResolutionAgent(StructuredLLM(original_provider), self.tools, AgentConfig(max_steps=2))
                expected = original.run(self.request)
                actual = self.run_script(script, max_steps=2)
                self.assertEqual(actual, expected)
                self.assertEqual(self.provider.requests, original_provider.requests)

    def test_proposal_behavioral_parity_ignoring_server_ids_and_timestamps(self):
        original = ResolutionAgent(StructuredLLM(baseline.ScriptedProvider([self.proposal(), baseline.FINAL])), self.tools)
        expected = original.run(self.request)
        actual = self.run_script([self.proposal(), baseline.FINAL])
        self.assertEqual(actual.status, expected.status)
        self.assertEqual(actual.steps_used, expected.steps_used)
        for result in (actual, expected):
            self.assertEqual(len(result.pending_proposal_ids), 1)
            proposal = result.history[1].result.data
            self.assertEqual(proposal.status, "PENDING_REVIEW")
            self.assertEqual(proposal.case_id, self.case.id)
            self.assertEqual(proposal.action, "refund")
            self.assertFalse(result.actions_executed)

    def test_invalid_state_and_public_request_cannot_inject_run_history(self):
        with self.assertRaises(ValidationError):
            GraphState(request=self.request, steps=-1)
        with self.assertRaises(ValidationError):
            GraphState(request=self.request, steps=True)
        with self.assertRaises(ValidationError):
            GraphState(request=self.request, arbitrary_function="approve")
        agent = GraphResolutionAgent(StructuredLLM(baseline.ScriptedProvider([])), self.tools)
        with self.assertRaises(ValidationError):
            agent.run({**self.request.model_dump(), "created": False, "history": []})

    def test_engine_failure_retains_last_completed_snapshot(self):
        for exception, code in ((GraphRecursionError("private"), "GRAPH_LIMIT"), (RuntimeError("private"), "GRAPH_ERROR")):
            with self.subTest(code=code):
                self.run_script([baseline.call("get_inventory", sku="LAP-1"), baseline.FINAL])
                snapshot = self.state.model_dump()
                snapshot["final_result"] = None
                def broken_stream(*args, **kwargs):
                    yield snapshot
                    raise exception
                with patch.object(self.agent._graph, "stream", side_effect=broken_stream):
                    result = self.agent.run(self.request)
                self.assertEqual(result.error, code)
                self.assertEqual(result.history, self.state.history)
                self.assertNotIn("private", result.model_dump_json())

    def test_environment_tracing_cannot_send_audit_to_langsmith(self):
        # The package is transitive only; even an inherited tracing setting must not cause I/O.
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-placeholder"}), \
                patch("langsmith.client.Client.create_run", side_effect=AssertionError("External tracing")) as create_run:
            result = self.run_script([baseline.FINAL])
            self.assertEqual(result.status, "INFORMATIONAL")
            create_run.assert_not_called()

    def test_engine_failure_after_write_preserves_staged_pending_proposal(self):
        self.run_script([self.proposal(), baseline.FINAL])
        snapshot = self.state.model_dump()
        event = self.state.history[1]
        snapshot.update(final_result=None, pending={}, created=False,
                        history=self.state.history[:1], staged_event=event, decision=event.decision, steps=1)
        def broken_stream(*args, **kwargs):
            yield snapshot
            raise GraphRecursionError("interrupted after execution")
        with patch.object(self.agent._graph, "stream", side_effect=broken_stream):
            result = self.agent.run(self.request)
        self.assertEqual(result.error, "GRAPH_LIMIT")
        self.assertEqual(result.pending_proposal_ids, (event.result.data.id,))
        self.assertEqual(len(result.history), 2)
        self.assertEqual(len(self.proposals.list_for_case(self.case.id)), 1)
