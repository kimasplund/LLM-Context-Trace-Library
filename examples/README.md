# Examples

Every example runs with no API key (each carries its own fake or scripted model), writes a
trace into the **current directory**, and prints the path. If the framework is missing it prints
the install command and exits 0. Run them from a scratch directory.

| Example | Extra | Install | Run | Look at |
|---------|-------|---------|-----|---------|
| `spans_demo.py` | (core) | `pip install lctl` | `python examples/spans_demo.py` | `lctl trace --tree spans_demo.lctl.json`: nested planner spans, two parallel workers, a caught error |
| `langchain_example.py` | `langchain` | `pip install "lctl[langchain] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/langchain_example.py` | `lctl trace --tree langchain_trace.lctl.json`: chain steps under a callback handler |
| `crewai_example.py` | `crewai` | `pip install "lctl[crewai] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/crewai_example.py` | `lctl trace --tree crewai_trace.lctl.json`: per-agent steps and the tool call |
| `openai_agents_example.py` | `openai-agents` | `pip install "lctl[openai-agents] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/openai_agents_example.py` | `lctl trace --tree openai_agents_trace.lctl.json`: agent run with a tool call |
| `pydantic_ai_example.py` | `pydantic-ai` | `pip install "lctl[pydantic-ai] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/pydantic_ai_example.py` | `lctl trace --tree pydantic_ai_trace.lctl.json`: traced agent run with token usage |
| `semantic_kernel_example.py` | `semantic-kernel` | `pip install "lctl[semantic-kernel] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/semantic_kernel_example.py` | `lctl trace --tree semantic_kernel_trace.lctl.json`: plugin function steps |
| `dspy_example.py` | `dspy` | `pip install "lctl[dspy] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/dspy_example.py` | `lctl trace --tree dspy_trace.lctl.json`: module forward spans |
| `llamaindex_example.py` | `llamaindex` | `pip install "lctl[llamaindex] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"` | `python examples/llamaindex_example.py` | `lctl trace --tree llamaindex_trace.lctl.json`: query, retriever and LLM spans |

`sample-chain.lctl.json` is a static sample chain used by the tests and docs.
