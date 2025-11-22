"""Tests for RTTM parsing, generation, and file formatting."""

import os
import tempfile
from src.diarization.rttm import RTTMSegment, parse_rttm, write_rttm

def test_rttm_segment_to_line():
    seg = RTTMSegment(uri="test_rec", speaker="SPEAKER_01", start=1.234, duration=4.567)
    expected = "SPEAKER test_rec 1 1.234 4.567 <NA> <NA> SPEAKER_01 <NA> <NA>"
    assert seg.to_line() == expected
    assert seg.end == 5.801

def test_parse_rttm_from_string():
    rttm_content = """
    SPEAKER an4 1 0.298 2.471 <NA> <NA> SPEAKER_A <NA> <NA>
    SPEAKER an4 1 3.163 1.983 <NA> <NA> SPEAKER_B <NA> <NA>
    """
    segments = parse_rttm(rttm_content)
    assert len(segments) == 2
    assert segments[0].uri == "an4"
    assert segments[0].speaker == "SPEAKER_A"
    assert segments[0].start == 0.298
    assert segments[0].duration == 2.471
    assert segments[1].speaker == "SPEAKER_B"
    assert segments[1].start == 3.163

def test_rttm_write_and_read_roundtrip():
    segments = [
        RTTMSegment(uri="conv_01", speaker="SPK_0", start=0.5, duration=2.0),
        RTTMSegment(uri="conv_01", speaker="SPK_1", start=3.0, duration=1.5),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        rttm_path = os.path.join(tmp_dir, "output.rttm")
        write_rttm(rttm_path, segments)
        assert os.path.exists(rttm_path)

        loaded = parse_rttm(rttm_path)
        assert len(loaded) == 2
        assert loaded[0].speaker == "SPK_0"
        assert loaded[0].start == 0.5
        assert loaded[1].speaker == "SPK_1"
        assert loaded[1].start == 3.0
