# Changelog

All notable changes to the LCTL (LLM Context Trace Library) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.1.0] - 2025-01-19

### Added

- Claude Code integration for multi-agent workflow tracing
- Comprehensive Claude Code tracing with advanced features (Task tool invocations, agent completions, tool calls, TodoWrite updates, skill invocations, MCP tool calls, git commits, user interactions)
- `lctl claude init` command to generate hook scripts for Claude Code integration
- `lctl claude validate` command to check hook installation
- `lctl claude status` command to show active session
- `lctl claude report` command to generate HTML reports
- `lctl claude clean` command to clean old traces
- `--chain-id` option to `lctl claude init` for custom chain identification
- Settings.json generation in `lctl claude init`
- React dashboard to VS Code extension
- Streaming API support
- RAG/retriever documentation to LangChain tutorial
- VS Code extension visualization enhancements
- VS Code extension screenshot to README
- DSPy and LlamaIndex integration documentation

### Changed

- Rebranded project to "LLM Context Trace Library"
- Updated all repository URLs to new project name
- Updated Python requirements
- Upgraded VS Code extension major dependencies
- Upgraded @vscode/test-cli from 0.0.4 to 0.0.12
- Upgraded vite to fix esbuild security vulnerability (CVE)
- Improved Claude Code hooks to use stdin JSON and propagate agent names

### Fixed

- Integration module exports and tutorial bugs
- Claude Code hooks to work from any subdirectory
- Comprehensive integration fixes: thread safety, memory leaks, and bugs
- Test collection errors for optional dependencies
- Comprehensive project review fixes
- Version consistency to 4.1.0

### Documentation

- Added comprehensive Claude Code integration documentation
- Added Claude Code tutorial (`docs/tutorials/claude-code-tutorial.md`)
- Updated task.md with license and history rewrite completion
- Added RAG/retriever documentation to LangChain tutorial
- Updated README with DSPy and LlamaIndex integrations
- Added VS Code extension screenshots

## [4.0.0] - 2025-01-15

Initial public release under AGPLv3 license.

### Added

- Core event sourcing system with Chain, Event, State, and ReplayEngine
- LCTLSession for recording events
- Time-travel replay functionality
- Chain comparison and diff tools
- Performance bottleneck analysis
- Confidence tracking for facts
- CLI commands: replay, stats, bottleneck, diff, trace, debug, dashboard
- Web dashboard with FastAPI
- Framework integrations:
  - LangChain (LCTLCallbackHandler, LCTLChain)
  - CrewAI (LCTLCrew, LCTLAgent, LCTLTask)
  - AutoGen/AG2 (LCTLAutogenCallback)
  - OpenAI Agents SDK (LCTLOpenAIAgentTracer, TracedAgent)
- VS Code extension for visualization
- JSON and YAML chain file format support
- Comprehensive test suite

[4.1.0]: https://github.com/kimasplund/LLM-Context-Trace-Library/compare/v4.0.0...v4.1.0
[4.0.0]: https://github.com/kimasplund/LLM-Context-Trace-Library/releases/tag/v4.0.0
