"""
Integration tests for the embeddings module.

These tests load the real multilingual-e5-base model — they are marked 'slow'
because:
  - First run downloads ~278MB from HuggingFace Hub (30-60s).
  - Subsequent runs load the model from local cache (~3-5s startup).

Run all tests:            pytest tests/ -v
Run only fast tests:      pytest tests/ -v -m "not slow"
Run only slow tests:      pytest tests/ -v -m slow

Rationale for testing the real model (vs mocking):
  - Validates the full stack (langchain-huggingface + sentence-transformers).
  - Catches config regressions (e.g. someone changing the model to one with
    a different dimension).
  - Empirically proves the multilingual behavior we rely on.
"""
import pytest

from app.core.embeddings import E5Embeddings, get_embeddings

# All tests in this module hit a real model → always slow.
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def embeddings():
    """
    Shared embeddings instance for all tests in this module.

    get_embeddings() is already a @lru_cache singleton, so this fixture is
    semantic sugar — it centralizes the dependency and makes future
    setup/teardown (e.g. cache_clear in CI) easier to wire in.
    """
    return get_embeddings()


def test_get_embeddings_returns_singleton():
    """The factory must return the same instance on repeated calls."""
    first = get_embeddings()
    second = get_embeddings()
    assert first is second, "get_embeddings() should return a cached singleton"


def test_embeddings_have_expected_dimension(embeddings):
    """
    multilingual-e5-base produces 768-dimensional vectors.

    Regression guard: if someone switches the configured model to one with
    a different dimension (e.g. all-MiniLM-L6-v2 → 384), this test fails
    loudly instead of silently breaking downstream retrieval.
    """
    vector = embeddings.embed_query("any text works here")
    assert len(vector) == 768, f"Expected 768 dims, got {len(vector)}"


def test_prefixes_produce_different_embeddings(embeddings):
    """
    The E5 prefixes must actually reach the model.

    Embedding the same text via embed_query vs embed_documents sends
    'query: X' and 'passage: X' to the model respectively. If the resulting
    vectors are identical, the prefix override is not working.

    We check for *any* difference (not a magnitude threshold) to keep the
    test robust across model versions.
    """
    text = "Il modello deve distinguere query da passage"
    query_vec = embeddings.embed_query(text)
    passage_vec = embeddings.embed_documents([text])[0]
    assert query_vec != passage_vec, (
        "Query and passage embeddings are identical — "
        "the E5 prefix override is not being applied."
    )


def test_multilingual_retrieval_semantic_similarity(embeddings):
    """
    Italian and English semantically equivalent sentences must embed closer
    to each other than to an unrelated Italian sentence.

    This is the end-to-end proof that multilingual-e5-base delivers on its
    promise for our use case (IT + EN documents). If this test ever fails,
    either the model was swapped or the prefixes are broken.
    """
    it_sentence = "Il gatto dorme sul divano"
    en_equivalent = "The cat sleeps on the couch"
    it_unrelated = "La ricetta della pasta al pomodoro"

    # Embed all three as queries — same prefix, fair comparison.
    v_it, v_en, v_other = embeddings.embed_query(it_sentence), \
                          embeddings.embed_query(en_equivalent), \
                          embeddings.embed_query(it_unrelated)

    # Vectors are L2-normalized (normalize_embeddings=True), so the dot
    # product equals cosine similarity. No need to import numpy.
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_cross_lingual = cosine(v_it, v_en)
    sim_unrelated = cosine(v_it, v_other)

    assert sim_cross_lingual > sim_unrelated, (
        f"Cross-lingual similarity ({sim_cross_lingual:.3f}) should exceed "
        f"unrelated similarity ({sim_unrelated:.3f}). "
        "Either the multilingual model or the normalization is misbehaving."
    )