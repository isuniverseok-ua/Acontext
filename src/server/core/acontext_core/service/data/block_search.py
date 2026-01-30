import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Tuple, cast

from ...constants import MetricTags
from ...telemetry.capture_metrics import capture_increment
from ...schema.orm import Block, BlockEmbedding
from ...schema.orm.block import PATH_BLOCK, CONTENT_BLOCK
from ...schema.utils import asUUID
from ...schema.result import Result
from ...llm.embeddings import get_embedding
from ...env import LOG


# TODO: add project_id to record
async def search_blocks(
    db_session: AsyncSession,
    space_id: asUUID,
    query_text: str,
    block_types: list[str],
    topk: int = 10,
    threshold: float = 0.8,
    fetch_ratio: float = 1.5,
) -> Result[List[Tuple[Block, float]]]:
    """
    Search for page and folder blocks using semantic vector similarity.

    Uses cosine distance on block embeddings with Python-side deduplication
    for optimal performance when blocks have multiple embeddings.

    Args:
        db_session: Database session
        space_id: Space to search within
        query_text: Search query text to embed and match against
        topk: Maximum number of unique blocks to return (default: 10)
        threshold: Maximum cosine distance threshold for matches (default: 1.0)
                  Range: 0.0 (identical) to 2.0 (opposite)
                  Typical good matches: 0.3-0.8

    Returns:
        Result containing list of (Block, distance) tuples sorted by similarity.
        Blocks are deduplicated - only the best match per block is returned.
        Lower distance = more similar.

    Example:
        >>> r = await search_path_blocks(
        ...     db_session=session,
        ...     space_id=space_uuid,
        ...     query_text="machine learning",
        ...     topk=5,
        ...     threshold=0.7
        ... )
        >>> if r.ok():
        ...     for block, distance in r.data:
        ...         print(f"{block.title}: {distance:.4f}")
    """
    # Generate query embedding
    r = await get_embedding([query_text], phase="query")
    if not r.ok():
        return r
    query_embedding = r.data.embedding[0]

    # Calculate distance using pgvector's cosine distance method
    # This uses the <=> operator for cosine distance in PostgreSQL
    distance = BlockEmbedding.embedding.cosine_distance(query_embedding).label(
        "distance"
    )

    # Fetch more than needed to account for blocks with multiple embeddings
    # Conservative estimate: 3x to ensure we get enough unique blocks
    fetch_limit = int(topk * fetch_ratio)

    # Build query - simple join without grouping for best performance
    query = (
        select(Block, distance)
        .join(BlockEmbedding, Block.id == BlockEmbedding.block_id)
        .where(
            Block.space_id == space_id,
            Block.type.in_(block_types),  # Only page and folder blocks
            Block.is_archived == False,  # Exclude archived blocks  # noqa: E712
            distance <= threshold,  # Apply distance threshold
        )
        .order_by(distance.asc())  # Best matches first
        .limit(fetch_limit)
    )

    # Execute query
    try:
        result = await db_session.execute(query)
        rows = result.all()

        # Deduplicate in Python: keep best (lowest distance) match per block
        # Since results are ordered by distance ASC, first occurrence is best
        seen: dict[asUUID, Tuple[Block, float]] = {}
        for row in rows:
            block, distance = cast(
                Tuple[Block, float], row
            )  # Unpack tuple: (Block, distance)
            if block.id in seen:
                continue
            seen[block.id] = (block, float(distance))

        # Get top-K unique blocks (already sorted by distance)
        results = list(seen.values())[:topk]

        # LOG.info(
        #     f"Search '{query_text[:50]}...' found {len(results)} unique blocks "
        #     f"(from {len(rows)} total embeddings)"
        # )
        return Result.resolve(results)

    except Exception as e:
        LOG.error(f"Error in search_path_blocks: {e}")
        return Result.reject(f"Vector search failed: {str(e)}")


async def search_path_blocks(
    db_session: AsyncSession,
    space_id: asUUID,
    query_text: str,
    topk: int = 10,
    threshold: float = 0.8,
    fetch_ratio: float = 1.5,
) -> Result[List[Tuple[Block, float]]]:
    return await search_blocks(
        db_session, space_id, query_text, list(PATH_BLOCK), topk, threshold, fetch_ratio
    )


async def search_content_blocks(
    db_session: AsyncSession,
    project_id: asUUID,
    space_id: asUUID,
    query_text: str,
    topk: int = 10,
    threshold: float = 0.8,
    fetch_ratio: float = 1.5,
) -> Result[List[Tuple[Block, float]]]:
    r = await search_blocks(
        db_session,
        space_id,
        query_text,
        list(CONTENT_BLOCK),
        topk,
        threshold,
        fetch_ratio,
    )
    if r.ok():
        asyncio.create_task(
            capture_increment(
                project_id=project_id,
                tag=MetricTags.new_experience_embedding_search,
            )
        )
    return r


# TODO: add project_id to record
async def search_sessions(
    db_session: AsyncSession,
    space_id: asUUID,
    query_text: str,
    topk: int = 10,
    threshold: float = 0.8,
    fetch_ratio: float = 5.0,  # Higher fetch ratio since many blocks might come from same session
) -> Result[List[asUUID]]:
    """
    Search for related sessions by finding SOP blocks that were digested from them.
    Blocks must have 'session_id' in their props.
    """
    # 1. Search for SOP blocks
    r = await search_blocks(
        db_session,
        space_id,
        query_text,
        [BLOCK_TYPE_SOP],
        topk=topk * 2,  # Fetch more blocks to ensure enough unique sessions
        threshold=threshold,
        fetch_ratio=fetch_ratio,
    )
    if not r.ok():
        return Result.reject(r.error)

    # 2. Extract unique session_ids with order preserved
    session_ids = []
    seen = set()

    for block, _ in r.data:
        if not block.props:
            continue
        sid_str = block.props.get("session_id")
        if not sid_str:
            continue

        if sid_str in seen:
            continue

        try:
            # Validate UUID format
            sid = asUUID(sid_str)
            seen.add(sid_str)
            session_ids.append(sid)
            if len(session_ids) >= topk:
                break
        except Exception:
            continue

    return Result.resolve(session_ids)
