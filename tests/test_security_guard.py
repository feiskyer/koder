from koder_agent.core.security import SecurityGuard


def test_validate_command_allows_dev_null_redirection():
    cmd = 'find . -name "SKILL.md" 2>/dev/null | head -1'
    assert SecurityGuard.validate_command(cmd) is None


def test_validate_command_blocks_other_dev_redirects():
    cmd = "echo test >/dev/sda"
    assert SecurityGuard.validate_command(cmd) is not None
