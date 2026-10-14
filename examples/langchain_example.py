"""LangChain example: a fake chat model traced by LCTLCallbackHandler (no API key).

Run it in a scratch directory; it writes ``langchain_trace.lctl.json`` there. View with:

    lctl trace --tree langchain_trace.lctl.json
"""

import sys

try:
    import langchain_core  # noqa: F401
except ImportError:
    print("LangChain is not installed. Install it with:")
    print(
        '  pip install "lctl[langchain] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate

from lctl.integrations.langchain import LCTLCallbackHandler, is_available


def main() -> None:
    if not is_available():
        print(
            'LCTL LangChain integration unavailable: pip install "lctl[langchain] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
        )
        sys.exit(0)

    handler = LCTLCallbackHandler(chain_id="langchain-example")
    prompt = ChatPromptTemplate.from_messages([("system", "Be brief."), ("human", "{question}")])
    chain = prompt | FakeListChatModel(responses=["Paris", "Berlin"])

    for question in ("Capital of France?", "Capital of Germany?"):
        answer = chain.invoke({"question": question}, config={"callbacks": [handler]})
        print(f"{question} -> {answer.content}")

    path = Path("langchain_trace.lctl.json")
    handler.export(str(path))
    print(f"Trace written to {path.resolve()} ({len(handler.chain.events)} events)")


if __name__ == "__main__":
    main()
