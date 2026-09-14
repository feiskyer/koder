import pytest

from koder_agent.cli import _build_cli_parser
from koder_agent.harness.cli.entrypoint import build_runtime_request, detect_first_arg


def test_runtime_request_classifies_help_and_empty_boot():
    assert build_runtime_request(["--help"]).mode == "help"
    assert build_runtime_request([]).mode == "interactive"


def test_runtime_request_skips_teammate_mode_when_detecting_prompt_mode():
    request = build_runtime_request(["--teammate-mode", "in-process", "-p", "/peers"])

    assert request.mode == "prompt"


@pytest.mark.parametrize(
    ("argv", "first_arg", "mode"),
    [
        (["--bare", "config", "validate"], "config", "subcommand"),
        (["--plugin-dir", "config", "doctor"], "doctor", "subcommand"),
        (["--image", "doctor", "describe this"], "describe this", "prompt"),
        (["-i", "config", "describe this"], "describe this", "prompt"),
        (["--plugin-dir=config", "doctor"], "doctor", "subcommand"),
        (["-ssession", "config", "validate"], "config", "subcommand"),
        (["--", "config", "validate"], None, "prompt"),
        (["--bare", "--", "config"], None, "prompt"),
        (["-pconfig"], None, "prompt"),
        (["--print=config"], None, "prompt"),
    ],
)
def test_runtime_routing_preserves_option_arity_and_explicit_prompts(argv, first_arg, mode):
    request = build_runtime_request(argv)

    assert request.first_arg == first_arg
    assert request.mode == mode
    _build_cli_parser(request.first_arg).parse_args(argv)


def test_runtime_routing_tracks_all_declared_value_options():
    parser = _build_cli_parser(None)
    for action in parser._actions:
        if action.nargs not in (None, "?"):
            continue
        for option in action.option_strings:
            assert detect_first_arg([option, "config", "doctor"]) == "doctor", option
