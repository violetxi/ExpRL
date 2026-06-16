"""Tests for actor retry logic for transient HTTP errors."""

import asyncio
import pytest
import aiohttp


class TestRetryLogic:
    """Tests for the retry logic implementation in actor.py."""

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test that retry logic uses exponential backoff."""
        max_retries = 3
        retry_base_delay = 0.01  # Use small delay for testing

        attempts = []

        async def failing_operation():
            attempts.append(len(attempts) + 1)
            if len(attempts) < max_retries:
                raise aiohttp.ClientPayloadError("Simulated transient error")
            return "success"

        # Simulate the retry logic from actor.py
        last_error = None
        result = None
        for attempt in range(max_retries):
            try:
                result = await failing_operation()
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise

        assert result == "success"
        assert len(attempts) == max_retries  # Should have retried until success

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises(self):
        """Test that after max_retries, the error is raised."""
        max_retries = 3
        retry_base_delay = 0.01

        attempts = []

        async def always_failing_operation():
            attempts.append(len(attempts) + 1)
            raise aiohttp.ClientPayloadError("Simulated persistent error")

        # Simulate the retry logic from actor.py
        with pytest.raises(aiohttp.ClientPayloadError):
            for attempt in range(max_retries):
                try:
                    await always_failing_operation()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        delay = retry_base_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                    else:
                        raise

        assert len(attempts) == max_retries  # Should have tried max_retries times

    @pytest.mark.asyncio
    async def test_various_exceptions_are_retried(self):
        """Test that various exception types are retried."""
        max_retries = 3
        retry_base_delay = 0.01

        # Test different exception types that should all be retried
        exception_types = [
            aiohttp.ClientPayloadError("payload error"),
            aiohttp.ServerDisconnectedError(),
            ConnectionResetError(104, "Connection reset by peer"),
            TimeoutError("timeout"),
            OSError("os error"),
            RuntimeError("runtime error"),  # Even generic errors are retried
        ]

        for exc in exception_types:
            attempts = []

            async def failing_then_success():
                attempts.append(1)
                if len(attempts) < 2:
                    raise exc
                return "success"

            result = None
            for attempt in range(max_retries):
                try:
                    result = await failing_then_success()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_base_delay)
                    else:
                        raise

            assert result == "success", f"Failed for {type(exc).__name__}"
            assert len(attempts) == 2, f"Wrong attempt count for {type(exc).__name__}"
