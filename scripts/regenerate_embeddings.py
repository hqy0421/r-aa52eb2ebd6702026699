#!/usr/bin/env python
"""
Rebuild the vector index from scratch for every article in the RSSidian database.

Unlike the daily backfill (``rssidian backfill-embeddings``), this script
re-embeds all articles rather than only the ones missing an embedding. Vectors
are committed in batches, so an interruption keeps the work already done; the
index is rebuilt from the database afterwards and written atomically.

A single unusable vector (wrong length, NaN/Inf, model failure) never aborts the
run: the article is skipped, left unflagged, and reported at the end so a later
run retries it.
"""

import os
import sys
import json
import logging
import math
import numbers
from typing import List, Optional, Dict, Any

from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rssidian.config import Config
from rssidian.models import get_db_session, Article
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2'
EMBEDDING_DIM = 768  # all-mpnet-base-v2 dimension


def coerce_embedding(raw: Any) -> Optional[List[float]]:
    """Return the vector as a list of finite floats, or None if unusable."""
    if raw is None:
        return None
    try:
        values = raw.tolist() if hasattr(raw, "tolist") else list(raw)
    except (TypeError, ValueError):
        return None
    if len(values) != EMBEDDING_DIM:
        return None
    vector = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return None
        component = float(value)
        if not math.isfinite(component):
            return None
        vector.append(component)
    return vector


def generate_embedding(model, text: str) -> List[float]:
    """Generate a validated embedding vector for text; raises ValueError if unusable."""
    # Truncate text if too long (most models have a max token limit)
    max_chars = 5000
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    embedding = coerce_embedding(model.encode(truncated_text))
    if embedding is None:
        raise ValueError("embedding model returned an unusable vector")
    return embedding


def save_index_atomically(index: AnnoyIndex, index_path: str) -> None:
    """Save an Annoy index via a temp file + atomic rename."""
    tmp_path = f"{index_path}.tmp.{os.getpid()}"
    try:
        index.save(tmp_path)
        os.replace(tmp_path, index_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def main():
    """Main function to regenerate embeddings."""
    # Load configuration
    config = Config()
    db_session = get_db_session(config.db_path)
    index_path = os.path.expanduser(config.annoy_index_path)

    try:
        # Count articles
        total_articles = db_session.query(Article).count()
        logger.info(f"Found {total_articles} articles in the database")

        # Load embedding model
        logger.info("Loading embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Get all articles
        articles = db_session.query(Article).order_by(Article.id).all()

        stats: Dict[str, int] = {
            "articles_processed": 0,
            "with_embedding": 0,
            "skipped": 0,
        }

        # Re-embed every article. Vectors are committed in batches so an
        # interruption does not discard work already done; the index itself is
        # rebuilt from the database afterwards.
        batch_size = 100
        logger.info(f"Processing {len(articles)} articles...")
        for i, article in enumerate(tqdm(articles, desc="Generating embeddings"), 1):
            text_to_embed = article.content
            if text_to_embed and article.summary:
                text_to_embed = article.summary + "\n\n" + text_to_embed

            if not text_to_embed:
                stats["skipped"] += 1
                stats["articles_processed"] += 1
                continue

            try:
                embedding = generate_embedding(embedding_model, text_to_embed)
            except Exception as e:
                logger.error(f"Error generating embedding for article {article.id}: {e}")
                # Drop a previous vector so it cannot masquerade as current.
                article.embedding_vector = None
                article.embedding_generated = False
                stats["skipped"] += 1
                stats["articles_processed"] += 1
                continue

            article.embedding_vector = json.dumps(embedding)
            article.embedding_generated = True
            stats["with_embedding"] += 1
            stats["articles_processed"] += 1

            if i % batch_size == 0:
                db_session.commit()

        db_session.commit()

        # Build the index from every article that now holds a usable vector.
        logger.info(f"Building vector index with {config.annoy_n_trees} trees...")
        vector_index = AnnoyIndex(EMBEDDING_DIM, config.annoy_metric)
        indexed = 0
        for article in db_session.query(Article).filter(
            Article.embedding_generated == True,  # noqa: E712
            Article.embedding_vector.isnot(None),
        ).all():
            vector = coerce_embedding(json.loads(article.embedding_vector))
            if vector is None:
                # Defensive: never let one bad row break the build.
                logger.warning(f"Skipping article {article.id}: stored vector is unusable")
                article.embedding_vector = None
                article.embedding_generated = False
                continue
            try:
                vector_index.add_item(article.id, vector)
            except (IndexError, TypeError, ValueError) as e:
                logger.warning(f"Skipping article {article.id}: Annoy rejected vector: {e}")
                article.embedding_vector = None
                article.embedding_generated = False
                continue
            indexed += 1

        db_session.commit()

        if indexed:
            vector_index.build(config.annoy_n_trees)
            save_index_atomically(vector_index, index_path)
            logger.info(f"Saved vector index with {indexed} vectors to {index_path}")
        else:
            logger.warning("No usable embeddings; existing index file left untouched")

        # Log results
        logger.info(f"Processed {stats['articles_processed']} articles:")
        logger.info(f"  With embeddings: {stats['with_embedding']}")
        logger.info(f"  Skipped (no text or unusable vector): {stats['skipped']}")

    except Exception as e:
        logger.error(f"Error during embedding regeneration: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
