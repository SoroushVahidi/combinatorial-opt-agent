"""Explicit capability opt-ins; ordinary test collection never probes the network."""
import socket
import pytest

CAPABILITIES = ("network", "models", "providers", "solvers", "research_data", "integration")

def pytest_addoption(parser):
    group = parser.getgroup("research capabilities")
    for capability in CAPABILITIES:
        group.addoption("--run-" + capability.replace("_", "-"), action="store_true", default=False,
                        help="Opt in to tests requiring " + capability.replace("_", " "))

def pytest_configure(config):
    for capability in CAPABILITIES:
        config.addinivalue_line("markers", "requires_" + capability + ": requires explicit capability opt-in")
    if not config.getoption("--run-network"):
        patch = pytest.MonkeyPatch()
        def blocked(*args, **kwargs):
            raise RuntimeError("Network disabled by offline test policy; use --run-network explicitly")
        original_connect = socket.socket.connect
        def connect(sock, address):
            if sock.family == socket.AF_UNIX:
                return original_connect(sock, address)
            return blocked()
        patch.setattr(socket, "create_connection", blocked)
        patch.setattr(socket.socket, "connect", connect)
        patch.setattr(socket.socket, "connect_ex", blocked)
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")
        patch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
        config._offline_patch = patch

def pytest_unconfigure(config):
    patch = getattr(config, "_offline_patch", None)
    if patch is not None:
        patch.undo()

def pytest_collection_modifyitems(config, items):
    for item in items:
        for capability in CAPABILITIES:
            option = "--run-" + capability.replace("_", "-")
            if item.get_closest_marker("requires_" + capability) and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason="Explicit opt-in required: " + option))

@pytest.fixture(scope="session")
def network_available(pytestconfig):
    """Whether network testing was requested; does not assert endpoint availability."""
    return pytestconfig.getoption("--run-network")
