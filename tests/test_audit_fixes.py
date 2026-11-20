"""Regression tests for the 2026-11-20 external-audit defects D1-D6."""
import re
from lens import __version__
from lens.server import answers_agree, origin_is_local

def test_version_consistency():
    m = re.search(r'^version = "([^"]+)"', open("pyproject.toml").read(), re.M)
    assert m and __version__ == m.group(1), "D3: __version__ vs metadata"

def test_agree_plain():
    assert answers_agree("Mars", "Mars.")
    assert answers_agree("Paris", "The capital is Paris, France.")

def test_agree_contradiction_D1():
    assert not answers_agree("Mars", "No, Mars is not the Red Planet.")
    assert not answers_agree("yes it does", "It does not.")
    assert not answers_agree("not Venus", "Venus is correct.")

def test_origin_wall_D4():
    assert origin_is_local(None)
    assert origin_is_local("http://127.0.0.1:8080")
    assert origin_is_local("http://localhost")
    assert not origin_is_local("https://evil.example")
    assert not origin_is_local("http://127.0.0.1.attacker.example")

def test_claims_partial_explicit_D2():
    src = open("lens/server.py").read()
    assert '"partial"' in src and '"claims_total"' in src, "D2: truncation must be explicit in response"

def test_retry_bounded_D5():
    src = open("lens/control.py").read()
    seg = src[src.index("resp, attempts = opened"):src.index("resp, attempts = opened") + 2200]
    assert "while True" not in seg and "attempts < 8" in seg, "D5: verified-answer loop must be bounded"
    assert "retry limit exceeded" in src

def test_ci_runs_tests_D6():
    wf = open(".github/workflows/publish.yml").read()
    assert "pytest tests/" in wf, "D6: publishing workflow must execute the test suite"

def test_badge_truth():
    rm, md = open("README.md").read(), open("docs/MODELS.md").read()
    badge = re.search(r"Behaviour%20packs-(\d+)", rm).group(1)
    doc = re.search(r"\*\*(\d+) models\*\*", md).group(1)
    assert badge == doc, f"badge {badge} vs docs {doc}"
