from __future__ import annotations

import socket

from engine.cli import _port_in_use


def test_port_with_a_listener_is_reported_busy():
    """A second viewer on a busy port does not fail on its own; it leaves the
    old one answering. The check has to catch that before starting."""
    with socket.socket() as listener:
        listener.bind(("localhost", 0))
        listener.listen()
        assert _port_in_use(listener.getsockname()[1])


def test_free_port_is_not_reported_busy():
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        port = probe.getsockname()[1]
    assert not _port_in_use(port)
