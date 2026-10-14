# Task: LLM Agent Framework Integrations

This task tracks the integration of various LLM agent frameworks into the LCTL (LLM Context Trace Library) project.

## Phase 1: Existing Integrations (Complete)

- [x] **DSPy Integration**
    - [x] Implement `lctl/integrations/dspy.py`
    - [x] Create tests in `tests/test_dspy.py`
    - [x] Update `docs/api.md`
- [x] **LlamaIndex Integration**
    - [x] Implement `lctl/integrations/llamaindex.py`
    - [x] Create tests in `tests/test_llamaindex.py`
    - [x] Update `docs/api.md`
- [x] **OpenAI Agents Integration**
    - [x] Implement `lctl/integrations/openai_agents.py`
    - [x] Create tests in `tests/test_openai_agents.py`
    - [x] Update `docs/api.md`

## Phase 2: PydanticAI Integration (Complete)

- [x] **PydanticAI Support**
    - [x] Install `pydantic-ai` dependency
    - [x] Implement `lctl/integrations/pydantic_ai.py`
        - [x] `LCTLPydanticAITracer` class
        - [x] `trace_agent` decorator/wrapper
        - [x] Handlers for agent runs and tool calls
    - [x] Create tests in `tests/test_pydantic_ai.py`
    - [x] Update `docs/api.md`

## Phase 3: Semantic Kernel Integration (Complete)

- [x] **Semantic Kernel Support**
    - [x] Install `semantic-kernel` dependency
    - [x] Implement `lctl/integrations/semantic_kernel.py`
        - [x] Kernel hooks/callbacks
        - [x] Function calling instrumentation
    - [x] Create tests in `tests/test_semantic_kernel.py`
    - [x] Update `docs/api.md`

## Phase 4: Finalization (Complete)

- [x] **Review and Polish**
    - [x] Run full test suite
    - [x] detailed coverage check
    - [x] Final documentation review
- [x] **Release**
    - [x] Bump version
    - [x] Create release notes

## Phase 5: Documentation & Tutorials (Complete)

- [x] **Tutorial Updates**
    - [x] Create `docs/tutorials/pydantic-ai-tutorial.md`
    - [x] Create `docs/tutorials/semantic-kernel-tutorial.md`
    - [x] Review `docs/tutorials/openai-agents-tutorial.md`
- [x] **Example Scripts**
    - [x] Add `examples/pydantic_ai_agent.py`
    - [x] Add `examples/semantic_kernel_agent.py`
    - [x] Add `examples/openai_agent.py`

## Phase 6: PydanticAI Enhancements (Complete)

- [x] **Data Capture**
    - [x] Research tool call interception in PydanticAI
    - [x] Implement `tool_start` / `tool_end` tracing
- [x] **Verification**
    - [x] Add tool call test cases to `tests/test_pydantic_ai.py`
    - [x] Add tool call test cases to `tests/test_pydantic_ai.py`
    - [x] Verify using example script

## Phase 7: AutoGen Modernization (Complete)

- [x] **Research**
    - [x] Inspect `autogen_core` CloudEvent structure
    - [x] Identify key event types for tracing (messages, tools, errors)
- [x] **Implementation**
    - [x] Enhance `LCTLAutogenHandler` to parse CloudEvents
    - [x] Verify async event handling
- [x] **Verification**
    - [x] Create `tests/test_autogen_modern.py`
    - [x] Verify with `autogen_core` example

## Phase 8: Integration Excellence (Complete)

- [x] **CrewAI Enhancement**
    - [x] Research `step_callback` behavior in async mode
    - [x] Implement robust async tracing (monkeypatch if needed)
    - [x] Verify with async test case
- [x] **Semantic Kernel Deep Dive**
    - [x] Explore richer event filters (Prompt rendering, Connectors)
    - [x] Enhance `LCTLSemanticKernelTracer`
- [x] **Final Review**
    - [x] Re-assess all modules for "Excellent" status

## Phase 9: Final Polish & Verification

- [x] **Standardization**
    - [x] Enforce `is_available()` in all integration modules
    - [x] Update `__all__` exports
- [x] **Comprehensive Testing**
    - [x] Run full test suite with coverage
    - [x] Verify no linting/syntax errors

## Phase 10: Closing Integration Gaps

- [x] **PydanticAI Streaming**
    - [x] Research `StreamedRunResult` structure
    - [x] Implement `trace_agent_stream` or wrap `run_stream`
    - [x] Verify with streaming test case
- [x] **Semantic Kernel Streaming**
    - [x] Investigate filter behavior with `invoke_stream`
    - [x] Implement generator wrapper in filter
    - [x] Verify with `research_sk_stream.py`
- [x] **CrewAI Token Usage**
    - [x] Research token usage availability in `Step` or `CrewOutput`
    - [x] Implement usage extraction
    - [x] Verify with `scan_crewai.py` (simulated check)

## Phase 11: Universal LLM Tracing

- [x] **Universal LLM Tracing** <!-- id: 11 -->
    - [x] **Core LCTL Support** <!-- id: 11.0 -->
        - [x] Add `LLM_TRACE` to `EventType` in `lctl/core/events.py` <!-- id: 11.0.1 -->
        - [x] Add `llm_trace` method to `LCTLSession` in `lctl/core/session.py` <!-- id: 11.0.2 -->
    - [x] **Research & Strategy** <!-- id: 11.1 -->
        - [x] Investigate PydanticAI `AgentRunResult` structure <!-- id: 11.1.1 -->
        - [x] Investigate CrewAI `LLM` class and interception points <!-- id: 11.1.2 -->
    - [x] **PydanticAI Implementation** <!-- id: 11.2 -->
        - [x] Implement `llm_trace` capture in `TracedAgent.run` <!-- id: 11.2.1 -->
        - [x] Implement `llm_trace` capture in `TracedStreamedRunResult` <!-- id: 11.2.2 -->
        - [x] Verify with tests <!-- id: 11.2.3 -->
    - [x] **CrewAI Implementation** <!-- id: 11.3 -->
        - [x] Create `LLMWrapper` or Interceptor for `crewai.llm.LLM.call` <!-- id: 11.3.1 -->
        - [x] Monkeypatch `crewai.llm.LLM` to use wrapper <!-- id: 11.3.2 -->
        - [x] Verify with tests <!-- id: 11.3.3 -->

## Phase 12: VS Code Extension Maturity

- [x] **Short Term: detailed Visualization** <!-- id: 12 -->
    - [x] **Event Parsing** <!-- id: 12.1 -->
        - [x] Update `LctlChainMetadata` interface to include `events` <!-- id: 12.1.1 -->
        - [x] Parse `events` in `chainProvider.ts` <!-- id: 12.1.2 -->
    - [x] **Dashboard Enhancement** <!-- id: 12.2 -->
        - [x] Update `webviewPanel.ts` HTML/CSS to support event list <!-- id: 12.2.1 -->
        - [x] Implement event item rendering (LLM, Tool, Step) <!-- id: 12.2.2 -->
    - [x] **Schema Validation** <!-- id: 12.3 -->
        - [x] Add version/schema check on load <!-- id: 12.3.1 -->
- [x] **Long Term: Real Python Integration** <!-- id: 13 -->
    - [x] **CLI Support** <!-- id: 13.1 -->
        - [x] Verify/Create `lctl replay` CLI entry point <!-- id: 13.1.1 -->
    - [x] **Extension Bridge** <!-- id: 13.2 -->
        - [x] Implement `PythonShell` or `cp.spawn` wrapper in extension <!-- id: 13.2.1 -->
        - [x] Connect `Replay` button to real CLI command <!-- id: 13.2.2 -->
        - [x] Stream CLI output to VS Code Output Channel (Used Terminal instead) <!-- id: 13.2.3 -->

## Phase 13: Extension Documentation
- [x] **README.md** <!-- id: 14 -->
    - [x] Create `vscode-lctl/README.md` with features and usage <!-- id: 14.1 -->
    - [x] Update install instructions to point to GitHub <!-- id: 14.2 -->
- [x] **User Manual** <!-- id: 15 -->
    - [x] Create `vscode-lctl/MANUAL.md` with detailed workflows <!-- id: 15.1 -->

## Phase 14: Project Hygiene (Complete)
- [x] **License Standardization** <!-- id: 16 -->
    - [x] Create root `LICENSE` (AGPLv3) <!-- id: 16.1 -->
    - [x] Update `pyproject.toml` and `README.md` <!-- id: 16.2 -->
    - [x] Update extension license <!-- id: 16.3 -->
    - [x] Rewrite git history to enforce AGPLv3 <!-- id: 16.4 -->
