"""LlamaIndex example: MockLLM/MockEmbedding with the Settings callback route (no API key).

Run it in a scratch directory; it writes ``llamaindex_trace.lctl.json`` there. View with:

    lctl trace --tree llamaindex_trace.lctl.json
"""

import sys

try:
    import llama_index.core  # noqa: F401
except ImportError:
    print("LlamaIndex is not installed. Install it with:")
    print(
        '  pip install "lctl[llamaindex] @ git+https://github.com/kimasplund/LLM-Context-Trace-Library"'
    )
    sys.exit(0)

from pathlib import Path

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

from lctl.integrations.llamaindex import LCTLLlamaIndexCallback, is_available


def main() -> None:
    if not is_available():
        print('LCTL LlamaIndex integration unavailable: pip install "lctl[llamaindex]"')
        sys.exit(0)

    Settings.llm = MockLLM()
    Settings.embed_model = MockEmbedding(embed_dim=8)

    callback = LCTLLlamaIndexCallback(chain_id="llamaindex-example")
    # Set before building the index: the index and its retriever read it at build time.
    Settings.callback_manager = callback.get_callback_manager()

    index = VectorStoreIndex.from_documents(
        [
            Document(text="Paris is the capital of France."),
            Document(text="Berlin is the capital of Germany."),
        ]
    )
    response = index.as_query_engine().query("What is the capital of France?")
    print(f"Query returned {len(str(response))} characters")

    path = Path("llamaindex_trace.lctl.json")
    callback.export(str(path))
    print(f"Trace written to {path.resolve()} ({len(callback.session.chain.events)} events)")


if __name__ == "__main__":
    main()
