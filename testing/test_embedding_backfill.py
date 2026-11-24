#!/usr/bin/env python
"""
Behavioral tests for embedding backfill / index rebuild resilience.

These exercise the real RSSProcessor against a temporary SQLite database and a
deterministic fake embedding model (no network, no model download). They cover
the failure the backfill is meant to survive:

- one article with an unusable vector must not stop later articles from being
  indexed and flagged;
- ``embedding_generated`` is only set for vectors that are actually searchable;
- progress committed before a crash survives, and a re-run resumes without
  recomputing completed work;
- re-running an already-complete backfill is a no-op.

Run: python testing/test_embedding_backfill.py
"""

import math
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from annoy import AnnoyIndex

from rssidian.core import EMBEDDING_DIM, RSSProcessor
from rssidian.models import Article, Feed, init_db


class FakeConfig:
    def __init__(self, index_path, n_trees=2):
        self.annoy_index_path = index_path
        self.annoy_n_trees = n_trees
        self.annoy_metric = "angular"
        self.similarity_threshold = 0.65
        self.excerpt_length = 100
        self.value_prompt_enabled = False


class FakeModel:
    """Deterministic embeddings; specific inputs produce unusable vectors."""

    def __init__(self, fail_on_call=None):
        self.calls = []
        self.fail_on_call = fail_on_call

    def encode(self, text):
        self.calls.append(text)
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("simulated embedding failure")
        if text.startswith("JUNK"):
            return [1.0] * (EMBEDDING_DIM - 1)  # wrong length -> Annoy rejects
        if text == "NAN":
            vector = [0.0] * EMBEDDING_DIM
            vector[0] = float("nan")
            return vector
        seed = (sum(ord(c) for c in text) % 11) + 1
        return [math.sin(seed + i) for i in range(EMBEDDING_DIM)]


def _index_has(index, article_id):
    try:
        return any(abs(x) > 1e-12 for x in index.get_item_vector(article_id))
    except IndexError:
        return False


def _setup():
    workdir = tempfile.mkdtemp(prefix="rssidian-backfill-")
    index_path = os.path.join(workdir, "annoy.idx")
    db = init_db(os.path.join(workdir, "test.db"))
    feed = Feed(title="Test Feed", url="https://example.com/feed")
    db.add(feed)
    db.commit()
    return workdir, index_path, db, feed


def _add(db, feed, key, content, processed=True):
    article = Article(
        feed_id=feed.id,
        title=f"Title {key}",
        url=f"https://example.com/{key}",
        guid=f"guid-{key}",
        content=content,
        processed=processed,
    )
    db.add(article)
    db.commit()
    return article


def test_bad_vector_does_not_block_others():
    _, index_path, db, feed = _setup()
    good = [_add(db, feed, i, f"good text {i}") for i in range(4)]
    bad_len = _add(db, feed, "jung", "JUNK-vector")
    bad_nan = _add(db, feed, "nan", "NAN")
    empty = _add(db, feed, "empty", None)

    processor = RSSProcessor(FakeConfig(index_path), db)
    processor._embedding_model = FakeModel()
    # batch_size=1 forces a rebuild after every article, the harshest case for
    # "one bad vector aborts the run".
    stats = processor.backfill_embeddings(batch_size=1)

    flags = {a.id: a.embedding_generated for a in db.query(Article).all()}
    assert stats["succeeded"] == 4, stats
    assert all(flags[a.id] for a in good), flags
    assert not flags[bad_len.id] and not flags[bad_nan.id], flags
    assert not flags[empty.id], flags

    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    assert index.load(index_path)
    assert all(_index_has(index, a.id) for a in good)
    assert not _index_has(index, bad_len.id)


def test_rerun_is_idempotent_and_skips_completed():
    _, index_path, db, feed = _setup()
    for i in range(3):
        _add(db, feed, i, f"good {i}")
    _add(db, feed, "jung2", "JUNK-again")

    first = RSSProcessor(FakeConfig(index_path), db)
    first._embedding_model = FakeModel()
    first.backfill_embeddings(batch_size=2)

    second = RSSProcessor(FakeConfig(index_path), db)
    model = FakeModel()
    second._embedding_model = model
    stats = second.backfill_embeddings(batch_size=2)

    # Completed articles must not be embedded again.
    assert not any(call.startswith("good") for call in model.calls), model.calls
    # Only the one unusable article is retried, and it still fails.
    assert stats == {"total": 1, "succeeded": 0, "failed": 1}, stats

    third = RSSProcessor(FakeConfig(index_path), db)
    third._embedding_model = FakeModel()
    # A fully embedded database has nothing to do: drop the one article that can
    # never be embedded, so every remaining article is complete.
    db.query(Article).filter(Article.embedding_generated == False).delete()  # noqa: E712
    db.commit()
    assert third.backfill_embeddings() == {"total": 0, "succeeded": 0, "failed": 0}


def test_crash_keeps_committed_progress_and_resumes():
    _, index_path, db, feed = _setup()
    articles = [_add(db, feed, i, f"resume {i}") for i in range(6)]

    processor = RSSProcessor(FakeConfig(index_path), db)
    processor._embedding_model = FakeModel()

    original_save = processor._save_index_atomically
    saves = {"count": 0}

    def failing_save(index):
        saves["count"] += 1
        if saves["count"] == 2:
            raise OSError("simulated disk failure")
        return original_save(index)

    processor._save_index_atomically = failing_save
    try:
        processor.backfill_embeddings(batch_size=2)
    except OSError:
        pass  # the simulated failure propagates, as a real one would

    flags = {a.id: a.embedding_generated for a in db.query(Article).order_by(Article.id)}
    first_batch = [a.id for a in articles[:2]]
    assert all(flags[i] for i in first_batch), flags
    assert not any(flags[a.id] for a in articles[2:]), flags

    # The atomically written index still holds exactly the first batch.
    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    assert index.load(index_path)
    assert all(_index_has(index, i) for i in first_batch)
    assert not _index_has(index, articles[2].id)

    # Resume: completed work is not recomputed, the rest is finished.
    resumed = RSSProcessor(FakeConfig(index_path), db)
    model = FakeModel()
    resumed._embedding_model = model
    stats = resumed.backfill_embeddings(batch_size=2)
    assert stats["total"] == 4, stats
    assert not any(call in {"resume 0", "resume 1"} for call in model.calls), model.calls
    assert all(a.embedding_generated for a in db.query(Article).all())

    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    assert index.load(index_path)
    assert all(_index_has(index, a.id) for a in articles)


def test_corrupt_stored_vector_is_repaired():
    _, index_path, db, feed = _setup()
    article = _add(db, feed, "corrupt", "recoverable text")
    article.embedding_vector = "not valid json"
    article.embedding_generated = True
    db.commit()

    processor = RSSProcessor(FakeConfig(index_path), db)
    model = FakeModel()
    processor._embedding_model = model
    stats = processor.backfill_embeddings()

    assert stats["succeeded"] == 1, stats
    assert "recoverable text" in model.calls
    db.refresh(article)
    assert article.embedding_generated is True
    assert processor._parse_stored_vector(article.embedding_vector) is not None


def test_missing_index_file_is_rebuilt_without_reembedding():
    _, index_path, db, feed = _setup()
    _add(db, feed, 0, "keep me")

    processor = RSSProcessor(FakeConfig(index_path), db)
    model = FakeModel()
    processor._embedding_model = model
    processor.backfill_embeddings()
    assert os.path.exists(index_path)

    os.remove(index_path)
    rebuild = RSSProcessor(FakeConfig(index_path), db)
    rebuild._embedding_model = FakeModel()
    stats = rebuild.backfill_embeddings()

    assert stats["total"] == 0, stats
    assert os.path.exists(index_path), "missing index must be rebuilt from stored vectors"


TESTS = [
    test_bad_vector_does_not_block_others,
    test_rerun_is_idempotent_and_skips_completed,
    test_crash_keeps_committed_progress_and_resumes,
    test_corrupt_stored_vector_is_repaired,
    test_missing_index_file_is_rebuilt_without_reembedding,
]


def main():
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print()
    if failures:
        print(f"{failures} test(s) failed")
        sys.exit(1)
    print("All embedding backfill tests passed")


if __name__ == "__main__":
    main()
