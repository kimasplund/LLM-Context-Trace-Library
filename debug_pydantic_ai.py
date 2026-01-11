try:
    import pydantic_ai
    print("import pydantic_ai: OK")
    from pydantic_ai import Agent, RunContext
    print("from pydantic_ai import Agent, RunContext: OK")
    from pydantic_ai.result import RunResult, Usage
    print("from pydantic_ai.result import RunResult, Usage: OK")
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart
    )
    print("from pydantic_ai.messages import ...: OK")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
