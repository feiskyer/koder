# Contributing to Koder

Welcome to Koder! We're excited that you want to contribute to this experimental AI coding assistant. This project is a learning-focused exploration of building advanced terminal-based AI agents, and we welcome contributions of all kinds.

## Project Philosophy

Koder is designed as both a functional tool and a learning playground for AI agent development. We value:

- **Experimentation**: Trying new approaches and learning from them
- **Clean Architecture**: Well-structured, maintainable Python code
- **Security First**: Robust validation and permission systems
- **Universal Compatibility**: Supporting multiple AI providers and use cases
- **Community Learning**: Sharing knowledge and growing together

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Git for version control
- API key for at least one AI provider (OpenAI, Anthropic, Google, etc.)

### Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/feiskyer/koder.git
   cd koder
   ```

2. **Set Up Environment**

   ```bash
   # Install dependencies
   uv sync

   # Configure AI provider (example: OpenAI)
   export OPENAI_API_KEY="your-api-key"
   export KODER_MODEL="gpt-4o"
   ```

3. **Verify Installation**

   ```bash
   # Test the CLI
   uv run koder "Hello, Koder!"

   # Run in interactive mode
   uv run koder
   ```

4. **Run Development Commands**

   ```bash
   # Code formatting
   uv run black .

   # Linting and fixes
   uv run ruff format
   uv run ruff check --fix

   # Error-only pylint check
   uv run pylint koder_agent/ --disable=C,R,W --errors-only
   ```

## Development Guidelines

### Code Style

We use automated tooling to maintain consistent code quality:

- **Black**: Code formatting (`uv run black .`)
- **Ruff**: Linting and import sorting (`uv run ruff format && uv run ruff check --fix`)
- **Pylint**: Additional error checking (`uv run pylint koder_agent/ --disable=C,R,W --errors-only`)
- **TUI scenarios**: Scenario verification (`uv run scripts/tmux_feature_scenarios.py --check`)

### Isolated Test Validation

Run primary unit and non-E2E integration tests against the Python/dependencies
behind your installed `koder` command, not a historical locked environment:

```bash
bash scripts/test.sh -q
bash scripts/test.sh --test tests/core -q
bash scripts/test.sh --test tests/core/test_goals_store.py -v
bash scripts/test.sh --runtime-report /tmp/koder-runtime.json --junitxml /tmp/koder-tests.xml -q
```

Each invocation re-reads the actual runtime's versions. Test tools are added in
a disposable overlay; the installed application is not changed. Every runtime
package and the Python version must still match before collection. The optional
`--runtime-report` records both inventories. Use `--runtime-python
/path/to/bin/python` to select another application install explicitly. Without
an identifiable installed runtime the command fails rather than silently using
the checkout's `.venv`. `run_isolated_tests.py` remains the lower-level runner for
explicit compatibility/CI environments, not a substitute for local runtime parity.
The shell entrypoint resolves `koder` before invoking `uv run` for Python: even
`uv run --no-project` can prepend the checkout's `.venv` to PATH. Do not prefix
`bash scripts/test.sh` with `uv run`.

`--test` is repeatable and accepts a file, directory or pytest node ID. Other
pytest options, such as `-k`, `-v` and `--tb=short`, are forwarded. Default test
selection is independent of the calling directory. Workspace, root configuration
and manual/live mode options are controlled by the runner.

Before importing pytest or collecting tests, the runner replaces inherited
provider and proxy configuration with a small runtime environment and creates a
temporary HOME, working directory and cache. Nested `uv` calls reuse the running
interpreter with synchronization and dotenv loading disabled. Python child
processes started through `subprocess` retain the guards even when a test passes
an explicit environment. Tests may still inject synthetic provider settings.

For subprocess tests, use the `python_child_environment` fixture and a temporary
`cwd` such as `tmp_path`. Changing HOME alone does not prevent dependencies from
searching the working directory for dotenv files during import. When asserting
exact program output through `uv run --no-project`, make the uv launcher quiet
so its own startup diagnostics are not confused with the program's stderr.

Python audit checks reject accesses to the original Koder/credential directories,
project-local `.koder`, and dotenv files outside the owned workspace. They also
reject remote socket connections/DNS and direct native Keychain/UI helpers.
Loopback connections remain available for controlled local fixtures. These are
accidental-access checks for trusted tests, **not an OS sandbox**: arbitrary native
programs, inherited file descriptors, deliberately disabled Python site
initialization, and malicious code are not contained by this runner.

Exit status is pytest's status. `--junitxml` writes outside the disposable HOME;
temporary test files are removed after success or failure. Use `--keep-workspace`
only when fixture files are needed for diagnosis; its printed directory is
retained. Tests requiring live providers or real terminal interaction are separate
acceptance runs, not silently treated as covered by this command.

Direct `uv run --no-env-file pytest ...` remains available for intentionally
unmanaged test runs, but does not provide the runner's collection-time profile,
credential or child-process guards.

### Supported-Python Test Contracts

Run behavioral tests on the minimum supported interpreter as well as the primary
development interpreter. Parsing source with an older grammar does not validate
runtime API availability or asynchronous exception behavior.

- Match `asyncio.wait_for` timeouts with `asyncio.TimeoutError`; it also names the
  built-in timeout exception on newer interpreters.
- If a test must keep the current task's identity, use the existing AnyIO
  `fail_after` context for its deadline. Moving the operation into `wait_for`
  creates a new task and can conceal task-local permission or ownership defects.
- To check cancellation messages across supported Python versions, use the
  `cancellation_observer` fixture around the application coroutine before
  scheduling it. Assert its recorded exception arguments as well as cancellation
  and completed cleanup. Older Tasks may drop the message when forwarding the
  exception to an awaiter; do not drop resource assertions or skip the test.

### Tool Module Staging

`ToolRegistry.register_module` stages module import and tool collection before
publication. Candidate code must not start child threads during those phases;
start runtime work only after the tool is published and invoked.

CPython 3.12+ supplies native thread-start audit events for this guard. Earlier
supported CPython versions use temporary checks at the standard `threading` and
`_thread` launch entrypoints under the module-load lock. The policy is local to
the candidate's thread, so unrelated threads may continue running and starting
their own work. The temporary entrypoints are restored after staging.

Staging is a cooperative module-publication contract, not a sandbox for hostile
Python code. The older-interpreter fallback cannot intercept native launch
functions retained before staging, arbitrary C extensions or code that replaces
the guard itself. Candidate modules must remain trusted application code.

### Wheel Validation

After building a wheel, check the exact output file against its source checkout:

```bash
uv build --wheel --out-dir dist
uv run --no-project --no-env-file python scripts/verify_wheel.py dist/koder-0.6.4-py3-none-any.whl --source . --json
```

Use the filename printed by the build if the project version changes. The
validator does not build, install, extract or import the application. It compares
the declared package inventory and bytes in both directions, including binary
assets; checks selected build metadata, console entrypoints, wheel RECORD hashes,
required assets and the source-declared Python syntax floor; and rejects unsafe,
duplicate, malformed or excessive archive members. Bytecode-cache exclusions and
validation limitations are included in the result. Supported Koder interpreters
use standard-library `tomllib`. The standalone verifier retains an optional
`tomli` fallback for inspecting older artifacts; that does not lower Koder's
runtime requirement.

An exit-0 static validation is not proof of runtime behavior on the minimum Python,
dependency compatibility, an atomic source snapshot or a CI run. Keep the source
unchanged during build and validation, and retain separate test/TUI evidence.

### TUI Validation Evidence

Validate the manifest before running a focused terminal scenario:

```bash
uv run scripts/tmux_feature_scenarios.py --check --strict-acceptance
uv run scripts/tmux_feature_scenarios.py --run slash_commands/loop --output-dir /tmp/koder-tui-evidence
```

Manifest validation does not exercise a real terminal. TUI behavior requires an
actual scenario run with its visible and persisted-state assertions.

Per-scenario setup, startup, turn, post-check, and cleanup exceptions are recorded
in `<suite>-<name>-error.txt` in the output directory. The record identifies the
failure stage and exception type. Other selected scenarios still run, and the
command exits nonzero if any scenario fails. Cleanup failures do not skip the
remaining cleanup attempts; keyboard interruption still stops the run.

Subprocess timeout and exit-error summaries omit command arguments and captured
output, which may contain credentials. Turn, pane, and post-assertion captures
remain separate. Each completed run replaces or clears its own exception record,
so an older error record does not survive a successful rerun.

### Code Quality Standards

1. **Type Hints**: Use type annotations for function parameters and return values
2. **Docstrings**: Document all public functions and classes
3. **Error Handling**: Implement proper exception handling with informative messages
4. **Security**: Follow SecurityGuard patterns for input validation
5. **Testing**: Add tests for new features (when applicable)

### Architecture Principles

1. **Separation of Concerns**: Keep tools, core logic, and UI separate
2. **Tool Registration**: Register SDK-facing tools in `koder_agent/tools/__init__.py:get_all_tools()`; keep execution logic in the owning harness or core service
3. **Security by Default**: All user inputs must be validated
4. **Provider Agnostic**: Support multiple AI providers through abstraction
5. **Rich UI**: Use Rich library for terminal interfaces

## Making Contributions

1. **Create an Issue** (for significant changes)

   - Describe the problem or feature request
   - Discuss the approach with maintainers
   - Get feedback before implementation

2. **Fork and Branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

3. **Develop and Test**

   - Write clean, documented code
   - Run `uv run black . && uv run ruff format && uv run ruff check --fix`
   - Test your changes thoroughly with the narrowest meaningful test first
   - For TUI behavior, validate scenarios with `uv run scripts/tmux_feature_scenarios.py --check` and run focused scenarios with `uv run scripts/tmux_feature_scenarios.py --run <name>`
   - Update documentation if needed

4. **Commit and Push**

   ```bash
   # Make atomic, descriptive commits
   git add .
   git commit -m "feat: add new tool for X functionality"

   # Push to your fork
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what you changed and why
   - Include testing information

## Security Considerations

### Security Requirements

1. **Input Validation**: All user inputs must be validated
2. **Command Safety**: Shell commands require SecurityGuard approval
3. **API Key Protection**: Never log or expose API keys
4. **Output Filtering**: Filter sensitive information from outputs
5. **Permission Checks**: Tool execution must check permissions

### Reporting Security Issues

For security vulnerabilities:

1. **DO NOT** create public issues
2. Email maintainers privately
3. Provide detailed reproduction steps
4. Allow time for responsible disclosure

## Community Guidelines

### Code of Conduct

- **Be Respectful**: Treat all community members with respect
- **Be Inclusive**: Welcome contributors of all backgrounds and skill levels
- **Be Collaborative**: Work together and help each other learn
- **Be Patient**: Remember that everyone is learning

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Reviews**: Ask for feedback on your contributions
