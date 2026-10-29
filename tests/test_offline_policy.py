"""Exercise collection-time blocking and capability gating in a child pytest run."""
import os
from pathlib import Path
import subprocess
import sys

def test_collection_is_offline_and_capabilities_are_opt_in(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "conftest.py").write_text((root / "conftest.py").read_text())
    (tmp_path / "test_policy.py").write_text("import socket, pytest\nwith pytest.raises(RuntimeError, match='Network disabled'):\n    socket.create_connection(('example.invalid', 443))\n@pytest.mark.requires_models\ndef test_model():\n    assert False, 'must be skipped'\ndef test_local():\n    assert True\n")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "1 passed, 1 skipped" in run.stdout
