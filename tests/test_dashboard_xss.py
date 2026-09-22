"""Output escaping in the graph dashboard, and input bounds on the payload.

The dashboard rendered graph data straight into innerHTML. That data comes
from trade records, and symbol/strategy/regime names on those records come
from inbound webhook alerts — so a crafted strategy name executed script in
the operator's browser, on a page that also displays account state.

One sink was worse than the rest: the neighbours list interpolated a node id
into an inline onclick, i.e. into a JavaScript context, so an id containing a
quote broke out of the handler entirely.
"""

import os
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TEMPLATE = Path(PROJECT_ROOT) / "shared" / "dashboard" / "templates" / "index.html"


@pytest.fixture(scope="module")
def template():
    return TEMPLATE.read_text(encoding="utf-8")


class TestEscapingHelper:
    def test_helper_exists(self, template):
        assert "function escapeHtml(" in template

    @pytest.mark.parametrize("char,entity", [
        ("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"),
        ('"', "&quot;"), ("'", "&#39;"),
    ])
    def test_helper_covers_every_dangerous_character(self, template, char, entity):
        body = template.split("function escapeHtml(")[1].split("\n}")[0]
        assert entity in body, f"escapeHtml does not encode {char!r}"

    def test_ampersand_is_replaced_first(self, template):
        """Otherwise &lt; becomes &amp;lt; and the output is double-encoded."""
        body = template.split("function escapeHtml(")[1].split("\n}")[0]
        assert body.index("&amp;") < body.index("&lt;")


class TestNoUnescapedInterpolation:
    """Every value interpolated into markup must go through escapeHtml."""

    SAFE_PREFIXES = ("escapeHtml", "pct", "barW", "wrPct", "latencyColor")

    def test_no_raw_interpolation_into_html(self, template):
        offenders = []
        for i, line in enumerate(template.splitlines(), 1):
            if not any(k in line for k in ("innerHTML", "html +=", "return '<")):
                continue
            for m in re.finditer(r"\+\s*([A-Za-z_][\w.\[\]()']*)\s*\+", line):
                expr = m.group(1)
                if expr.startswith(self.SAFE_PREFIXES) or ".toFixed(" in expr:
                    continue
                offenders.append(f"line {i}: {expr}")
        assert not offenders, "unescaped interpolation:\n" + "\n".join(offenders)

    def test_toast_escapes_both_fields(self, template):
        """Every showToast caller passes WebSocket data straight through."""
        block = template.split("function showToast(")[1].split("\n}")[0]
        assert "escapeHtml(title)" in block
        assert "escapeHtml(body)" in block

    def test_no_inline_onclick_built_from_data(self, template):
        """The neighbours list injected a node id into a JS context."""
        assert "onclick=\"network.selectNodes" not in template
        assert "data-node-id=" in template
        assert "addEventListener('click'" in template


class TestPayloadInputBounds:
    """Defence in depth: bound the values on the way in as well."""

    def _payload(self, **kw):
        from tradingview.webhooks.webhook_server import AlertPayload
        base = dict(symbol="AAPL", action="buy", price=100.0)
        base.update(kw)
        return AlertPayload(**base)

    def test_ordinary_strategy_name_is_accepted(self):
        assert self._payload(strategy="trend_following").strategy == "trend_following"

    def test_script_tag_in_strategy_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._payload(strategy="<script>alert(1)</script>")

    def test_overlong_strategy_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._payload(strategy="a" * 200)

    def test_script_tag_in_regime_is_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._payload(regime="<img src=x onerror=alert(1)>")

    def test_newlines_in_message_are_flattened(self):
        """An embedded newline would otherwise forge a log line."""
        p = self._payload(message="real line\nFAKE 2026-01-01 CRITICAL forged")
        assert "\n" not in p.message
        assert "\r" not in p.message

    def test_message_length_is_bounded(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._payload(message="x" * 5000)

    def test_message_still_allows_normal_punctuation(self):
        text = "Crossed above 20-EMA; RSI 62.4 (bullish)."
        assert self._payload(message=text).message == text
