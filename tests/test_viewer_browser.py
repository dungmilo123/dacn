"""Real-browser acceptance for the Dash replay viewer (S05/T03, R011, D020).

The offline suites (``test_viewer.py`` / ``test_viewer_app.py``) contract-test
the projection, the Dash layout, and the callback reducers. This module is the
complementary *rendered* proof: it serves the real app on an ephemeral loopback
port and drives Chromium so Cytoscape and Plotly actually hydrate and a user can
step the replay.

The smoke is **opt-in** via ``DFL_RUN_BROWSER_TESTS=1`` so the ordinary
``uv run pytest -q`` suite never needs an installed browser binary. When the
variable is set the test is *not* skipped: a missing Chromium install or a
hydration failure surfaces as a real failure rather than a silent skip.

Preconditions (documented in README.md):

    uv sync
    uv run playwright install chromium
    DFL_RUN_BROWSER_TESTS=1 uv run pytest -q tests/test_viewer_browser.py --browser chromium

DOM selectors are anchored only to the stable ``viewer-*`` component IDs and
accessible status text declared in ``dfl_sim.viewer``; none depend on Dash's
generated renderer internals. Canvas-pixel semantics stay contract-tested in
the offline suites, so this module records hydration and round/UI transitions
rather than brittle pixel assertions.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from werkzeug.serving import make_server

from dfl_sim import create_viewer_app

from helpers import make_viewer_trace

pytestmark = [
    pytest.mark.browser,
    pytest.mark.skipif(
        os.environ.get("DFL_RUN_BROWSER_TESTS") != "1",
        reason=(
            "real-browser smoke is opt-in; set DFL_RUN_BROWSER_TESTS=1 after "
            "'uv run playwright install chromium'"
        ),
    ),
]

#: The fixture trace is the representative two-round/three-node replay.
_ROUND_ZERO_STATUS = "round 0 of 1: 4 directed message(s) from 3 sender(s)"
_ROUND_ONE_STATUS = "round 1 of 1: 4 directed message(s) from 3 sender(s)"


@dataclass
class ViewerServer:
    """A live loopback URL plus the owning server for teardown assertions."""

    url: str
    server: object
    thread: threading.Thread


@pytest.fixture
def viewer_server(tmp_path: Path) -> ViewerServer:
    """Serve the real Dash app on an ephemeral 127.0.0.1 port.

    The app is exactly what ``dfl view`` builds (validated projection, Dash
    layout, callbacks); only the transport is swapped for a lifecycle-managed
    Werkzeug server. The ``finally`` block guarantees shutdown even when a
    browser assertion raises, and asserts the serving thread actually exited.
    The fixture binds loopback only and never accepts input from the page.
    """

    trace = make_viewer_trace(tmp_path / "browser_trace")
    app = create_viewer_app(trace)
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    assert host == "127.0.0.1"
    try:
        yield ViewerServer(f"http://{host}:{port}/", server, thread)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive(), "viewer server thread did not shut down"


def _goto(page: Page, url: str) -> None:
    """Navigate, then wait for Dash/Plotly to hydrate the network canvas."""

    page.goto(url, wait_until="load")
    # Cytoscape draws its canvases only after the layout JSON is fetched and the
    # component mounts; this is the deterministic hydration barrier.
    expect(page.locator("#viewer-network canvas").first).to_be_visible(
        timeout=30_000
    )
    expect(page.locator("#viewer-loss-graph .main-svg").first).to_be_visible(
        timeout=30_000
    )
    expect(page.locator("#viewer-accuracy-graph .main-svg").first).to_be_visible(
        timeout=30_000
    )


def test_fixture_server_is_loopback_reachable(viewer_server: ViewerServer) -> None:
    """The lifecycle-managed fixture really serves the app while it is active."""

    assert viewer_server.url.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(viewer_server.url, timeout=10) as response:
        assert response.status == 200
        assert "dfl replay" in response.read().decode("utf-8")


def test_viewer_hydrates_network_legend_and_charts(
    page: Page, viewer_server: ViewerServer
) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))

    _goto(page, viewer_server.url)

    summary = page.locator("#viewer-trace-summary")
    expect(summary).to_be_visible()
    expect(summary).to_contain_text("browser_trace")
    expect(summary).to_contain_text("schema v1")
    expect(summary).to_contain_text("3 node(s)")
    expect(summary).to_contain_text("[1 malicious]")

    legend = page.locator("#viewer-legend")
    expect(legend).to_be_visible()
    expect(legend).to_contain_text("benign node")
    expect(legend).to_contain_text("malicious node")

    # The rendered network has real Cytoscape canvases (nodes/edges drawn).
    expect(page.locator("#viewer-network canvas").first).to_be_visible()
    # Both Plotly figures rendered their main SVG and every node series label.
    for graph_id in ("#viewer-loss-graph", "#viewer-accuracy-graph"):
        graph = page.locator(graph_id)
        expect(graph.locator(".main-svg").first).to_be_visible()
        for node_label in ("node 0", "node 1", "node 2"):
            expect(graph).to_contain_text(node_label)

    # The initial actual round + directed-message status is shown and announced.
    label = page.locator("#viewer-round-label")
    expect(label).to_have_text("round 0 of 1")
    assert label.get_attribute("aria-live") == "polite"
    status = page.locator("#viewer-round-status")
    expect(status).to_have_text(_ROUND_ZERO_STATUS)
    assert status.get_attribute("aria-live") == "polite"

    # Current-round cursor/value: the selected node's round-0 loss is annotated.
    expect(page.locator("#viewer-loss-graph")).to_contain_text("node 0: 0.900")

    assert errors == [], errors


def test_round_stepping_updates_state_and_clamps_at_bounds(
    page: Page, viewer_server: ViewerServer
) -> None:
    _goto(page, viewer_server.url)

    label = page.locator("#viewer-round-label")
    status = page.locator("#viewer-round-status")
    prev_button = page.locator("#viewer-prev-button")
    next_button = page.locator("#viewer-next-button")

    expect(label).to_have_text("round 0 of 1")
    expect(status).to_have_text(_ROUND_ZERO_STATUS)

    # Next advances to the next *actual* round and updates accessible state,
    # including the current-round metric annotation.
    next_button.click()
    expect(label).to_have_text("round 1 of 1")
    expect(status).to_have_text(_ROUND_ONE_STATUS)
    expect(page.locator("#viewer-loss-graph")).to_contain_text("node 0: 0.700")

    # Clicking Next at the terminal round cannot leave the valid range.
    next_button.click()
    expect(label).to_have_text("round 1 of 1")
    expect(status).to_have_text(_ROUND_ONE_STATUS)

    # Previous returns to the earlier round and its metrics.
    prev_button.click()
    expect(label).to_have_text("round 0 of 1")
    expect(status).to_have_text(_ROUND_ZERO_STATUS)
    expect(page.locator("#viewer-loss-graph")).to_contain_text("node 0: 0.900")

    # Clicking Previous at the first round also clamps.
    prev_button.click()
    expect(label).to_have_text("round 0 of 1")
    expect(status).to_have_text(_ROUND_ZERO_STATUS)


def test_node_selection_emphasizes_series_in_browser(
    page: Page, viewer_server: ViewerServer
) -> None:
    _goto(page, viewer_server.url)

    # Select node 2 through the real dropdown; the loss chart re-renders with
    # node 2 emphasized and its current value annotated. dcc.Dropdown renders a
    # virtualized react-select, so open the control then pick the option row.
    page.locator("#viewer-node-selector .Select-control").click()
    page.locator(".VirtualizedSelectOption", has_text="node 2").first.click()

    expect(page.locator("#viewer-loss-graph")).to_contain_text(
        "node 2 emphasized"
    )
    expect(page.locator("#viewer-loss-graph")).to_contain_text("node 2: 1.500")
    # Emphasizing one series keeps every node available.
    for node_label in ("node 0", "node 1", "node 2"):
        expect(page.locator("#viewer-loss-graph")).to_contain_text(node_label)
