"""Tests for the streams module, particularly handling of corrupted stream files we encountered with runs like conf/proof_qwen3-4b-thinking_v00.00.yaml"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from pipelinerl.streams import (
    FileStreamReader,
    FileStreamWriter,
    SingleStreamSpec,
    stream_dir,
    stream_file,
)


@pytest.fixture
def temp_exp_path():
    """Create a temporary directory for the experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_backend():
    """Reset the streams backend before and after each test."""
    import pipelinerl.streams as streams_module
    original_backend = streams_module._backend
    streams_module._backend = None
    yield
    streams_module._backend = original_backend


class TestFileStreamReaderCorruptedLine:
    """
    Tests for FileStreamReader handling of corrupted lines.

    These tests reproduce the exact failure pattern we saw in the 
    streams/weight_update_request/0/0/0.jsonl logs where garbage data
    like "54244}\\n" appeared mid-file, causing the reader to fail.
    """

    def test_read_valid_stream(self, temp_exp_path):
        """Baseline test: reading a valid stream should work."""
        import pipelinerl.streams as streams_module
        streams_module._backend = "files"

        spec = SingleStreamSpec(exp_path=temp_exp_path, topic="test_valid", instance=0, partition=0)

        # Write valid data
        with FileStreamWriter(spec, mode="w") as writer:
            writer.write({"kind": "test", "value": 1})
            writer.write({"kind": "test", "value": 2})
            writer.write({"kind": "test", "value": 3})

        # Read it back
        results = []
        with FileStreamReader(spec) as reader:
            for item in reader.read():
                results.append(item)
                if len(results) >= 3:
                    break

        assert len(results) == 3
        assert results[0]["value"] == 1
        assert results[1]["value"] == 2
        assert results[2]["value"] == 3

    def test_skip_corrupted_line_and_continue(self, temp_exp_path):
        """
        Reproduce the exact failure and verify recovery.

        From the stream logs, the corruption pattern was:
        - Valid JSON lines
        - Garbage line: "54244}\\n"
        - More valid JSON lines

        The reader should skip the garbage and continue reading.
        """
        import pipelinerl.streams as streams_module
        streams_module._backend = "files"

        # Create the exact corruption pattern from logs
        _file_dir = stream_dir(temp_exp_path, "weight_update_request", instance=0, partition=0)
        os.makedirs(_file_dir, exist_ok=True)
        file_path = stream_file(_file_dir, 0)

        with open(file_path, "w") as f:
            # Write several valid entries first
            for i in range(10):
                f.write(f'{{"kind":"samples_processed","samples_processed":{i*100},"timestamp":1764875551.565069}}\n')

            # Write a weight update success
            f.write('{"kind":"weight_update_success","version":4096,"timestamp":1764875551.565069}\n')

            # The exact corruption pattern from logs
            f.write('54244}\n')  # This is the garbage line

            # Valid data after corruption
            f.write('{"kind":"samples_processed","samples_processed":4115,"timestamp":1764875551.5654244}\n')
            f.write('{"kind":"samples_processed","samples_processed":4124,"timestamp":1764875551.5654244}\n')

        spec = SingleStreamSpec(
            exp_path=temp_exp_path,
            topic="weight_update_request",
            instance=0,
            partition=0
        )

        results = []
        with FileStreamReader(spec, max_retries=1) as reader:  # skip_corrupted=True by default, fast retries for test
            for item in reader.read():
                results.append(item)
                if len(results) >= 13:  # Expected count: 10 + 1 + 2 valid lines
                    break

        # Verify we read ALL valid data including after the corruption
        # 10 samples_processed + 1 weight_update_success + 2 more samples_processed = 13
        assert len(results) == 13

        # Verify the data before corruption is correct
        assert results[10]["kind"] == "weight_update_success"
        assert results[10]["version"] == 4096

        # Verify we recovered and read the data after corruption
        assert results[11]["kind"] == "samples_processed"
        assert results[11]["samples_processed"] == 4115
        assert results[12]["kind"] == "samples_processed"
        assert results[12]["samples_processed"] == 4124

    def test_corrupted_line_raises_when_skip_disabled(self, temp_exp_path):
        """
        With skip_corrupted=False, the reader should fail on corrupted data (original behavior).
        """
        import pipelinerl.streams as streams_module
        streams_module._backend = "files"

        _file_dir = stream_dir(temp_exp_path, "weight_update_request", instance=0, partition=0)
        os.makedirs(_file_dir, exist_ok=True)
        file_path = stream_file(_file_dir, 0)

        with open(file_path, "w") as f:
            for i in range(10):
                f.write(f'{{"kind":"samples_processed","samples_processed":{i*100},"timestamp":1764875551.565069}}\n')
            f.write('{"kind":"weight_update_success","version":4096,"timestamp":1764875551.565069}\n')
            f.write('54244}\n')  # Garbage
            f.write('{"kind":"samples_processed","samples_processed":4115,"timestamp":1764875551.5654244}\n')

        spec = SingleStreamSpec(
            exp_path=temp_exp_path,
            topic="weight_update_request",
            instance=0,
            partition=0
        )

        results = []
        with FileStreamReader(spec, skip_corrupted=False, max_retries=1) as reader:
            with pytest.raises(json.JSONDecodeError):
                for item in reader.read():
                    results.append(item)
                    if len(results) >= 12:  # More than expected before corruption (11)
                        break

        # Should have read all valid data before corruption
        assert len(results) == 11  # 10 samples_processed + 1 weight_update_success
