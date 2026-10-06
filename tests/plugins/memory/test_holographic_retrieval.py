"""Tests for FactRetriever FTS5 query sanitization.

These tests cover the fix where raw natural-language queries passed to
FTS5 MATCH were AND-joined by default, dropping recall to zero on any
multi-word prose query. The sanitizer drops stopwords and OR-joins the
remaining content tokens as phrase literals.
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")  # retrieval module imports numpy indirectly

from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore

# ---------------------------------------------------------------------------
# _sanitize_fts_query — unit tests (no DB required)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query,expected_tokens",
    [
        # stopwords dropped
        ("what happened with the deployment rollback", {"happened", "deployment", "rollback"}),
        # single content word passes through
        ("compaction", {"compaction"}),
        # all stopwords → falls back to raw
        ("the and of", None),  # None = sentinel for fallback-to-raw
        # empty string → empty output
        ("", ""),
        # FTS5 operator characters never survive; a hyphenated word becomes a phrase of the
        # words the unicode61 tokenizer indexed ("length" then "probe"), not "lengthprobe"
        ("context: length-probe", {"context", "length probe"}),
        ("hermes-bridge", {"hermes bridge"}),
        ("[hermes-bridge]", {"hermes bridge"}),
        ("llama-swap config", {"llama swap", "config"}),
        ("path/to.file_name", {"path to file name"}),
        # trailing punctuation stripped by tokenizer
        ("hello, world!", {"hello", "world"}),
    ],
)
def test_sanitize_fts_query_extracts_content_tokens(query, expected_tokens):
    result = FactRetriever._sanitize_fts_query(query)

    if expected_tokens == "":
        assert result == ""
        return

    if expected_tokens is None:
        # Pathological case: all stopwords — should fall back to raw query
        assert result == query
        return

    # OR-joined phrase literals: `"tok1" OR "tok2" OR ...`
    # Extract the tokens between quotes, order-independent.
    import re
    matches = re.findall(r'"([^"]+)"', result)
    assert set(matches) == expected_tokens, f"got {result!r}"

# ---------------------------------------------------------------------------
# Integration test — actually run _fts_candidates against an in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def retriever_with_facts(tmp_path):
    """MemoryStore seeded with a few facts for retrieval tests."""
    db_path = tmp_path / "test_facts.db"
    store = MemoryStore(str(db_path))
    store.add_fact(
        content="The Thursday deployment rollback failed because of stale migration state.",
        category="project",
    )
    store.add_fact(
        content="Compaction settings tuned to 0.85 threshold.",
        category="tool",
    )
    store.add_fact(
        content="Venice.ai advertises availableContextTokens inside model_spec.",
        category="tool",
    )
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()

def test_prefetch_recovers_prose_query(retriever_with_facts):
    """A natural-language query should now match the relevant fact.

    Before the sanitizer fix, 'what happened with the deployment rollback'
    returned zero hits because FTS5 required every token to co-occur.
    """
    results = retriever_with_facts.search(
        "what happened with the deployment rollback"
    )
    assert len(results) >= 1
    # The top hit should be the deployment rollback fact
    assert "deployment rollback" in results[0]["content"].lower()

@pytest.fixture
def retriever_with_hyphenated_facts(tmp_path):
    """Facts whose key terms contain in-word punctuation, as tags and content."""
    store = MemoryStore(str(tmp_path / "hyphen_facts.db"))
    store.add_fact(content="[hermes-bridge] Leftover steer text is returned as pending_steer.",
                   category="project", tags="hermes-bridge")
    store.add_fact(content="The llama-swap proxy serves local models.", category="tool")
    store.add_fact(content="Bridge tournaments use duplicate scoring.", category="general")
    retriever = FactRetriever(store=store)
    yield retriever
    store.close()


@pytest.mark.parametrize("query,needle", [
    ("hermes-bridge", "pending_steer"),
    ("[hermes-bridge]", "pending_steer"),
    ("llama-swap", "llama-swap proxy"),
])
def test_hyphenated_query_matches_its_fact(retriever_with_hyphenated_facts, query, needle):
    """A hyphenated query finds the fact that contains the same hyphenated term.

    unicode61 indexes 'hermes-bridge' as the words 'hermes' and 'bridge'; deleting the '-'
    from the query searched for 'hermesbridge', which no fact contains, so FTS returned nothing.
    """
    candidates = retriever_with_hyphenated_facts._fts_candidates(query, None, 0.0, 10)
    assert any(needle in c["content"] for c in candidates), [c["content"] for c in candidates]
    assert needle in retriever_with_hyphenated_facts.search(query)[0]["content"]


def test_hyphenated_query_matches_the_phrase_not_either_word(retriever_with_hyphenated_facts):
    """The words of a hyphenated term must appear together: 'hermes-bridge' is not 'bridge'."""
    contents = [c["content"] for c in retriever_with_hyphenated_facts._fts_candidates("hermes-bridge", None, 0.0, 10)]
    assert not any("tournaments" in c for c in contents), contents


@pytest.mark.parametrize("query", [
    'foo-bar', 'a:b test', '(xx', 'xx)', 'foo*', '^start', 'foo+bar', 'NEAR(aa bb)', 'xx OR yy', 'C++ code',
    'e-mail!', "it's here", '¿qué-tal?', 'tab\there', 'say "hi"', 'x"y"z', 'back\\slash', 'under_score',
])
def test_sanitized_query_is_always_valid_fts5(retriever_with_hyphenated_facts, query):
    """Any query with a content word becomes a MATCH expression SQLite accepts (no raw fallback)."""
    sanitized = FactRetriever._sanitize_fts_query(query)
    assert sanitized != query
    retriever_with_hyphenated_facts.store._conn.execute(
        "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?", [sanitized]).fetchall()


# ---------------------------------------------------------------------------
# Loop-invariant encode hoists (perf) — search/probe/related must encode
# constant vectors ONCE per call, not once per candidate/row.
# encode_text/encode_atom are deterministic (SHA-256 counter blocks), so the
# hoisted vectors are bit-identical to the per-iteration values they replace.
# ---------------------------------------------------------------------------

from plugins.memory.holographic import holographic as hrr

def test_encode_functions_are_deterministic():
    """Soundness premise of the hoists: same input -> identical vector."""
    import numpy as np

    assert np.array_equal(hrr.encode_text("deploy target", 1024),
                          hrr.encode_text("deploy target", 1024))
    assert np.array_equal(hrr.encode_atom("__hrr_role_content__", 1024),
                          hrr.encode_atom("__hrr_role_content__", 1024))
