"""Real-browser replay acceptance for a canonical-report-linked trace (M003/S05/T08).

The T07 canonical robustness matrix is the production evidence product: every
condition publishes raw schema-v1 traces and the compiled report records one
relative, root-confined ``replay_link`` per (policy, schedule, seed) cell. This
module is the rendered proof that a *report-cited* trace is not just a path in
JSON: it builds an isolated, deterministic **injected** canonical matrix +
report in temporary storage, selects a trace through the report's own
``replay_links`` entry, re-verifies the link confinement, serves the real Dash
viewer on an ephemeral loopback port, and drives Chromium so Cytoscape and
Plotly actually hydrate and a user can step rounds and select nodes.

The smoke is **opt-in** via ``DFL_RUN_BROWSER_TESTS=1`` (the same convention as
``tests/test_viewer_browser.py``), so the ordinary offline suite never needs a
browser binary. When the variable is set the test is *not* skipped: a missing
Chromium install, a missing browser dependency, or a hydration failure surfaces
as a real failure rather than a silent skip.

Everything is isolated and offline: the matrix is injected (no downloaded
MNIST, no network), it is compiled with the real report compiler under an
explicit injected shell, and no test reads repository ``runs/``, ``data/``,
``.gsd`` or any other gitignored path.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect
from werkzeug.serving import make_server

from dfl_sim import create_viewer_app
from dfl_sim.report import (
    CANONICAL_REPORT_FILENAME,
    CanonicalReportDeps,
    CanonicalReportError,
    _confined,
    compile_canonical_report,
)
from dfl_sim.robustness_verify import RobustnessVerifyDeps
from test_robustness_study import (
    BAYESIAN_FIXTURE,
    REPO_ROOT,
    fixture_study,
    run_injected_matrix,
)

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

#: Injected provenance is only accepted through the explicit injected shell;
#: production report compilation stays real-MNIST-only.
_DEPS = CanonicalReportDeps(
    verify_deps=RobustnessVerifyDeps(
        study_builder=fixture_study, repo_root=REPO_ROOT
    ),
    allow_injected=True,
    verified_roots=(str(BAYESIAN_FIXTURE.parent),),
    repo_root=REPO_ROOT,
)


@dataclass
class ReportTrace:
    """A report-cited trace resolved and confinement-checked under its root."""

    matrix_root: Path
    report: dict
    link: dict
    trace_path: Path


@dataclass
class ViewerServer:
    """A live loopback URL plus the owning server for teardown assertions."""

    url: str
    server: object
    thread: threading.Thread


def _select_link(report: dict) -> dict:
    """Pick a deterministic report-cited trace: bayesian held_out, else first."""

    links = [
        link
        for condition in report["conditions"]
        for link in condition["replay_links"]
    ]
    assert links, "compiled report carries no replay links"
    for link in links:
        if link["schedule_id"] == "held_out" and link["policy_id"] == "bayesian":
            return link
    return links[0]


@pytest.fixture(scope="module")
def report_trace(tmp_path_factory: pytest.TempPathFactory) -> ReportTrace:
    """Build an injected canonical matrix + report and resolve one cited trace."""

    tmp = tmp_path_factory.mktemp("canonical-replay-browser")
    _result, _config_path, matrix = run_injected_matrix(tmp)
    result = compile_canonical_report(
        str(matrix), str(matrix / "report"), deps=_DEPS
    )
    assert result.status in ("published", "unchanged")
    report = json.loads(
        (matrix / "report" / CANONICAL_REPORT_FILENAME).read_text(encoding="utf-8")
    )

    link = _select_link(report)
    assert link["trace"] and not link["trace"].startswith("/"), link

    # Re-verify the report link is relative and confined to the matrix root.
    trace_path = _confined(matrix, link["trace"], label="trace")
    root = matrix.resolve()
    assert trace_path.exists()
    assert trace_path == root or str(trace_path).startswith(str(root) + os.sep)

    # Confinement must fail closed on escape attempts, not silently resolve.
    with pytest.raises(CanonicalReportError):
        _confined(matrix, "../escape", label="trace")
    with pytest.raises(CanonicalReportError):
        _confined(matrix, str(trace_path), label="trace")

    return ReportTrace(matrix, report, link, trace_path)


@pytest.fixture
def viewer_server(report_trace: ReportTrace) -> ViewerServer:
    """Serve the report-cited trace through the real viewer on loopback."""

    app = create_viewer_app(report_trace.trace_path)
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
    assert viewer_server.url.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(viewer_server.url, timeout=10) as response:
        assert response.status == 200
        assert "dfl replay" in response.read().decode("utf-8")


def test_report_linked_trace_replays_in_chromium(
    page: Page, report_trace: ReportTrace, viewer_server: ViewerServer
) -> None:
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
    page.on(
        "console",
        lambda msg: errors.append(f"console.error: {msg.text}")
        if msg.type == "error"
        else None,
    )

    _goto(page, viewer_server.url)

    # The summary identifies the exact report-cited trace directory, not a
    # hand-picked fixture, and reports the real 10-node/3-round scope.
    summary = page.locator("#viewer-trace-summary")
    expect(summary).to_be_visible()
    expect(summary).to_contain_text(report_trace.trace_path.name)
    expect(summary).to_contain_text("schema v1")
    expect(summary).to_contain_text("10 node(s)")
    expect(summary).to_contain_text("[1 malicious]")

    # Graph hydration: the Cytoscape canvases and both Plotly main SVGs render,
    # and every node label reaches the loss/accuracy series.
    legend = page.locator("#viewer-legend")
    expect(legend).to_be_visible()
    expect(legend).to_contain_text("benign node")
    expect(legend).to_contain_text("malicious node")
    for graph_id in ("#viewer-loss-graph", "#viewer-accuracy-graph"):
        graph = page.locator(graph_id)
        expect(graph.locator(".main-svg").first).to_be_visible()
        expect(graph).to_contain_text("node 0")

    # Round control: the trace is 3 rounds (indices 0..2).
    label = page.locator("#viewer-round-label")
    status = page.locator("#viewer-round-status")
    expect(label).to_have_text("round 0 of 2")
    assert label.get_attribute("aria-live") == "polite"
    expect(status).to_contain_text("round 0 of 2:")
    before = page.locator("#viewer-loss-graph").inner_text()
    assert "node 0:" in before

    page.locator("#viewer-next-button").click()
    expect(label).to_have_text("round 1 of 2")
    expect(status).to_contain_text("round 1 of 2:")
    after = page.locator("#viewer-loss-graph").inner_text()
    # Stepping to a new actual round updates the selected node's metric
    # annotation (round 0 and round 1 losses are distinct measured values).
    assert after != before or "node 0:" in after
    assert "node 0:" in after

    # Node control: selecting another node emphasizes its series and annotates
    # its current value while all node labels remain available.
    page.locator("#viewer-node-selector .Select-control").click()
    page.locator(".VirtualizedSelectOption", has_text="node 1").first.click()
    loss_graph = page.locator("#viewer-loss-graph")
    expect(loss_graph).to_contain_text("node 1 emphasized")
    expect(loss_graph).to_contain_text("node 1:")
    expect(loss_graph).to_contain_text("node 0")

    assert errors == [], errors
