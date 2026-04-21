"""Tests for token bucket rate limiter."""

import asyncio

import pytest

from gateway.middleware.rate_limit import TokenBucket


class TestTokenBucket:
    """Test the token bucket rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_allows_under_capacity(self):
        bucket = TokenBucket(rate=10.0, capacity=10)
        for _ in range(10):
            assert await bucket.acquire() is True

    @pytest.mark.asyncio
    async def test_rejects_over_capacity(self):
        bucket = TokenBucket(rate=1.0, capacity=2)
        assert await bucket.acquire() is True
        assert await bucket.acquire() is True
        assert await bucket.acquire() is False

    @pytest.mark.asyncio
    async def test_refills_over_time(self):
        bucket = TokenBucket(rate=10.0, capacity=1)
        assert await bucket.acquire() is True
        assert await bucket.acquire() is False

        await asyncio.sleep(0.15)  # Wait for ~1.5 tokens to refill
        assert await bucket.acquire() is True

    @pytest.mark.asyncio
    async def test_capacity_is_max(self):
        bucket = TokenBucket(rate=100.0, capacity=3)
        await asyncio.sleep(0.1)  # Would add 10 tokens but capped at 3
        count = 0
        for _ in range(10):
            if await bucket.acquire():
                count += 1
        assert count == 3

    @pytest.mark.asyncio
    async def test_retry_after(self):
        bucket = TokenBucket(rate=1.0, capacity=1)
        await bucket.acquire()
        retry = bucket.retry_after
        assert retry > 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Ensure thread-safety with concurrent requests."""
        bucket = TokenBucket(rate=100.0, capacity=10)
        results = await asyncio.gather(*[bucket.acquire() for _ in range(20)])
        acquired = sum(1 for r in results if r)
        assert acquired == 10  # Exactly capacity
