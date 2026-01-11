# Task: LLM Agent Framework Integrations

This task tracks the integration of various LLM agent frameworks into the LCTL (LLM Context Transfer Layer) project.

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

## Phase 3: Semantic Kernel Integration (Next)

- [ ] **Semantic Kernel Support**
    - [ ] Install `semantic-kernel` dependency
    - [ ] Implement `lctl/integrations/semantic_kernel.py`
        - [ ] Kernel hooks/callbacks
        - [ ] Function calling instrumentation
    - [ ] Create tests in `tests/test_semantic_kernel.py`
    - [ ] Update `docs/api.md`

## Phase 4: Finalization

- [ ] **Review and Polish**
    - [ ] Run full test suite
    - [ ] detailed coverage check
    - [ ] Final documentation review
- [ ] **Release**
    - [ ] Bump version
    - [ ] Create release notes
