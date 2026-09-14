"""Tool orchestration helpers.

``ToolOrchestrator`` from ``tools/orchestration.py`` is an experimental read/write
batching utility, not part of the active agent execution path. The openai-agents
SDK owns tool scheduling; this utility's serial-write policy must not be assumed
to protect SDK-run tools.

Note: the former ``ToolEngine`` class was removed because it had no runtime call
site — tools are exposed to the agent via the SDK ``@function_tool`` decorators
(see ``get_all_tools`` in ``tools/__init__.py``), not through ``ToolEngine``.
"""

from .orchestration import ToolOrchestrator


def get_orchestrator() -> ToolOrchestrator:
    """Get a ToolOrchestrator instance for concurrent read-only batching.

    Example of standalone use (not the production SDK path)::

        orchestrator = get_orchestrator()
        results = await orchestrator.execute_batch(
            calls=[
                {"tool": "read_file", "args": {"path": "file1.py"}},
                {"tool": "read_file", "args": {"path": "file2.py"}},
            ],
            executor=executor_callable,
        )
    """
    return ToolOrchestrator()
