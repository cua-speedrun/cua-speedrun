"""Real HTTP file transfers and the placement invariant; no simulated VM."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
import threading

import pytest

from cua_speedrun.client import Computer
from cua_speedrun.evaluation_runtime import InstanceInfrastructureError
from cua_speedrun.remote.run import require_same_region


@pytest.mark.parametrize("expected,actual", [
    (None, None), (None, "new-york"), ("new-york", None),
    ("new-york", "CHI"), ("", ""), ("us-east", "new-york"),
])
def test_region_must_be_known_and_identical(expected, actual):
    with pytest.raises(InstanceInfrastructureError, match="same-region"):
        require_same_region(expected, actual)


@pytest.mark.parametrize("region", ["new-york", "CHI", "eastus"])
def test_exact_actual_region_is_preserved(region):
    assert require_same_region(region, region) == region


@pytest.mark.parametrize("profile", [False, True])
def test_profile_real_persistent_http_file_transfer(tmp_path, profile):
    data = bytes(range(256)) * 2000
    (tmp_path / "payload.bin").write_bytes(data)

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = os.environ.get("CS_FAST_IO_PROFILE")
    os.environ["CS_FAST_IO_PROFILE"] = str(profile).lower()
    computer = Computer(f"http://127.0.0.1:{server.server_port}", fast_io=True)
    try:
        for index in range(2):
            response = computer._request("GET", "payload.bin")
            assert response.content == data
            timing = computer.last_request_timing
            assert timing["response_bytes"] == len(data)
            assert timing["attempts"] == 1
            if profile:
                details = timing["transport_profile"]
                assert details["connection_reused"] is (index > 0)
                assert details["http_version"] == 11
                assert details["headers_ready_sec"] > 0
                assert details["body_receive_sec"] > 0
                assert sum(details[key] for key in (
                    "headers_ready_sec", "hook_sec", "body_receive_sec"
                )) == pytest.approx(timing["round_trip_sec"])
            else:
                assert "transport_profile" not in timing
    finally:
        computer.close()
        server.shutdown()
        server.server_close()
        thread.join()
        if previous is None:
            os.environ.pop("CS_FAST_IO_PROFILE", None)
        else:
            os.environ["CS_FAST_IO_PROFILE"] = previous
