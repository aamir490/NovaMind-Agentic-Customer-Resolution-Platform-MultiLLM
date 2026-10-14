"""Explicit local LangGraph orchestration; Phase 8 remains the reference implementation."""

import json
from uuid import UUID

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context
from pydantic import Field

from .agent import AgentConfig, AgentDecision, AgentRequest, AgentResult, AuditEntry, Contract, Finish, ToolCall, agent_instructions
from .llm import LLMMessage, LLMRequest, StructuredLLM
from .tools import LocalTools
from .conversations import (ConversationService, ConversationError,
                            begin_memory, finish_memory)
from .security import SecurityError
from .guardrails import GuardrailError, check_payload
from .observability import observed


class GraphState(Contract):
    request: AgentRequest
    memory_started: bool = False
    steps: int = Field(default=0, strict=True, ge=0, le=32)
    messages: tuple[LLMMessage, ...] = ()
    allowed_names: tuple[str, ...] = ()
    decision: ToolCall | Finish | None = None
    staged_event: AuditEntry | None = None
    history: tuple[AuditEntry, ...] = ()
    pending: dict[UUID, bool] = Field(default_factory=dict)
    created: bool = False
    error: str | None = None
    final_result: AgentResult | None = None
    node_path: tuple[str, ...] = ()


class GraphResolutionAgent:
    def __init__(self, llm: StructuredLLM, tools: LocalTools, config: AgentConfig | None = None,
                 *, memory: ConversationService | None = None) -> None:
        self._llm = llm
        self._tools = tools
        self._config = config or AgentConfig()
        self._memory = memory
        graph = StateGraph(GraphState)
        for name in ("load_case", "reason", "execute_tool", "record_result", "finalize", "fail_safely"):
            graph.add_node(name, getattr(self, "_" + name))
        graph.add_edge(START, "load_case")
        graph.add_conditional_edges("load_case", lambda s: "fail_safely" if s.error else "reason",
                                    {name: name for name in ("fail_safely", "reason")})
        graph.add_conditional_edges("reason", self._after_reason,
                                    {name: name for name in ("fail_safely", "finalize", "execute_tool")})
        graph.add_edge("execute_tool", "record_result")
        graph.add_conditional_edges("record_result", lambda s: "fail_safely" if s.error else "reason",
                                    {name: name for name in ("fail_safely", "reason")})
        graph.add_edge("finalize", END)
        graph.add_edge("fail_safely", END)
        self._graph = graph.compile()  # Conversation text is separate from graph state persistence.

    @staticmethod
    def _update(state: GraphState, node: str, **changes) -> dict:
        changes["node_path"] = (*state.node_path, node)
        # Validate outgoing updates too; LangGraph validates state on entry to nodes.
        validated = GraphState.model_validate({
            **{key: getattr(state, key) for key in GraphState.model_fields}, **changes,
        })
        return {key: getattr(validated, key) for key in changes}

    def run(self, request: AgentRequest) -> AgentResult:
        return self.run_state(request).final_result

    @observed("agent.graph")
    def run_state(self, request: AgentRequest) -> GraphState:
        """Run fresh state only; callers cannot inject history, counters, or resume a write."""
        state = GraphState(request=AgentRequest.model_validate(request))
        # One reasoning step uses at most three nodes, plus load and terminal nodes.
        # The business step budget is enforced separately before every LLM invocation.
        with tracing_context(enabled=False):
            try:
                for snapshot in self._graph.stream(
                    {"request": state.request}, stream_mode="values",
                    config={"recursion_limit": 3 * self._config.max_steps + 5, "callbacks": []},
                ):
                    state = GraphState.model_validate(snapshot)
            except GraphRecursionError:
                state = GraphState.model_validate({**state.model_dump(), "error": "GRAPH_LIMIT"})
            except Exception:
                state = GraphState.model_validate({**state.model_dump(), "error": "GRAPH_ERROR"})
        if state.final_result is None:
            state = GraphState.model_validate({
                **state.model_dump(), **self._fail_safely(state),
            })
        if state.memory_started:
            try:
                finish_memory(self._memory, state.request.conversation_id, state.request.case_id,
                              state.final_result.customer_response)
            except ConversationError as failure:
                event = AuditEntry(step=state.steps, error=failure.code)
                state = GraphState.model_validate({**state.model_dump(), "error": failure.code,
                    "history": (*state.history, event),
                    "final_result": self._result(state, (*state.history, event), failure.code)})
        return state

    def _load_case(self, state: GraphState) -> dict:
        call = ToolCall(kind="tool", name="get_case", arguments={"case_id": str(state.request.case_id)})
        try:
            result = self._tools.invoke(call.name, call.arguments)
        except Exception:
            return self._update(state, "load_case", staged_event=AuditEntry(step=0, decision=call, error="TOOL_ERROR"),
                                error="TOOL_ERROR")
        event = AuditEntry(step=0, decision=call, result=result)
        if not result.ok:
            return self._update(state, "load_case", staged_event=event, error=result.error.code)
        try:
            check_payload(state.request.model_dump(mode="json"))
            conversation_id, memory_context = begin_memory(
                self._memory, state.request.conversation_id, state.request.case_id, state.request.message)
        except (ConversationError, SecurityError, GuardrailError) as failure:
            return self._update(state, "load_case", history=(event,), error=failure.code,
                                staged_event=AuditEntry(step=0, error=failure.code))
        request = AgentRequest(**{**state.request.model_dump(), "conversation_id": conversation_id})
        descriptions = self._tools.describe()
        instructions = agent_instructions(descriptions)
        messages = [LLMMessage(role="system", content=instructions)]
        if memory_context is not None:
            messages.append(LLMMessage(role="user", content=memory_context))
        messages.append(LLMMessage(role="user", content=json.dumps({
            "request": request.model_dump(mode="json"), "case_result": result.model_dump(mode="json"),
        })))
        return self._update(state, "load_case", history=(event,), request=request,
                            memory_started=conversation_id is not None,
                            allowed_names=tuple(item.name for item in descriptions), messages=tuple(messages))

    @staticmethod
    def _after_reason(state: GraphState) -> str:
        if state.error:
            return "fail_safely"
        return "finalize" if isinstance(state.decision, Finish) else "execute_tool"

    def _reason(self, state: GraphState) -> dict:
        if state.steps >= self._config.max_steps:
            return self._update(state, "reason", error="STEP_LIMIT")
        step = state.steps + 1
        try:
            answer = self._llm.generate(LLMRequest(messages=state.messages,
                                       max_output_tokens=self._config.max_output_tokens), AgentDecision)
        except Exception:
            return self._update(state, "reason", steps=step, error="LLM_ERROR",
                                staged_event=AuditEntry(step=step, error="LLM_ERROR"))
        if not answer.ok:
            return self._update(state, "reason", steps=step, error=answer.code,
                                staged_event=AuditEntry(step=step, llm_failure=answer))
        return self._update(state, "reason", steps=step, decision=answer.data.decision)

    def _execute_tool(self, state: GraphState) -> dict:
        call = state.decision
        error = None
        if not isinstance(call, ToolCall):
            error = "INVALID_RESPONSE"
        elif call.name not in state.allowed_names:
            error = "UNKNOWN_TOOL"
        elif call.name == "create_resolution_proposal":
            if state.created:
                error = "PROPOSAL_LIMIT"
            else:
                try:
                    same_case = UUID(str(call.arguments.get("case_id"))) == state.request.case_id
                except ValueError:
                    same_case = False
                if not same_case:
                    error = "CASE_MISMATCH"
        if error:
            return self._update(state, "execute_tool", error=error,
                                staged_event=AuditEntry(step=state.steps, decision=call, error=error))
        try:
            # The Phase 5 dispatcher validates its existing input models BEFORE calling services.
            result = self._tools.invoke(call.name, call.arguments)
        except Exception:
            return self._update(state, "execute_tool", error="TOOL_ERROR",
                                staged_event=AuditEntry(step=state.steps, decision=call, error="TOOL_ERROR"))
        return self._update(state, "execute_tool",
                            staged_event=AuditEntry(step=state.steps, decision=call, result=result))

    def _record_result(self, state: GraphState) -> dict:
        event = state.staged_event
        if event is None:
            return self._update(state, "record_result", error="GRAPH_ERROR")
        changes = {"history": (*state.history, event), "staged_event": None}
        result = event.result
        if state.error:
            return self._update(state, "record_result", **changes)
        if result is None:
            return self._update(state, "record_result", **changes, error="GRAPH_ERROR")
        if not result.ok:
            return self._update(state, "record_result", **changes, error=result.error.code)
        call = event.decision
        if call.name in ("create_resolution_proposal", "get_proposal_status"):
            record = result.data.model_dump(mode="json")
            if record["case_id"] != str(state.request.case_id):
                return self._update(state, "record_result", **changes, error="CASE_MISMATCH")
            changes["pending"] = {**state.pending, UUID(record["id"]): record["status"] == "PENDING_REVIEW"}
            if call.name == "create_resolution_proposal":
                changes["created"] = True
                if record["status"] != "PENDING_REVIEW":
                    return self._update(state, "record_result", **changes, error="INVALID_PROPOSAL_STATUS")
        changes["messages"] = (*state.messages,
            LLMMessage(role="assistant", content=AgentDecision(decision=call).model_dump_json()),
            LLMMessage(role="user", content=json.dumps({"tool_result": result.model_dump(mode="json")})),
        )
        if state.steps >= self._config.max_steps:
            changes["error"] = "STEP_LIMIT"
        return self._update(state, "record_result", **changes)

    @staticmethod
    def _result(state: GraphState, history: tuple[AuditEntry, ...], error: str | None = None) -> AgentResult:
        ids = tuple(key for key, value in state.pending.items() if value)
        message = "Informational results only. No refund, return, replacement, or inventory action was executed."
        if ids:
            message = ("Proposal(s) recorded as PENDING_REVIEW require separate human review: "
                       + ", ".join(map(str, ids)) + ". No action was executed.")
        if error:
            message = "The review could not be completed. " + message
        return AgentResult(status="FAILED" if error else "HUMAN_REVIEW_REQUIRED" if ids else "INFORMATIONAL",
                           conversation_id=state.request.conversation_id if state.memory_started else None,
                           customer_response=message, steps_used=state.steps, history=history,
                           pending_proposal_ids=ids, error=error)

    def _finalize(self, state: GraphState) -> dict:
        history = (*state.history, AuditEntry(step=state.steps, decision=state.decision))
        return self._update(state, "finalize", history=history, final_result=self._result(state, history))

    def _fail_safely(self, state: GraphState) -> dict:
        error = state.error or "GRAPH_ERROR"
        # If the engine failed between execution and recording, retain a completed
        # tool outcome (especially a pending write) without executing it again.
        if state.staged_event and state.staged_event.result and state.staged_event.result.ok:
            recovered = GraphState.model_validate({**state.model_dump(), "error": None})
            state = GraphState.model_validate({**recovered.model_dump(), **self._record_result(recovered)})
        history = (*state.history, state.staged_event) if state.staged_event else state.history
        return self._update(state, "fail_safely", history=history, staged_event=None, error=error,
                            final_result=self._result(state, history, error))
