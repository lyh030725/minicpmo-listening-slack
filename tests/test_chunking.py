import numpy as np

from slackbench.dataset import full_second_chunks, realtime_chunks


def test_full_second_chunks_drops_tail():
    audio = np.arange(25, dtype=np.float32)
    chunks = list(full_second_chunks(audio, sample_rate=10))
    assert len(chunks) == 2
    assert chunks[0][0] == 0
    assert chunks[1][0] == 1
    assert len(chunks[0][1]) == 10
    assert len(chunks[1][1]) == 10
    assert chunks[1][1][-1] == 19


def test_realtime_chunks_keeps_tail_and_adds_silence():
    audio = np.arange(25, dtype=np.float32)
    chunks = list(realtime_chunks(audio, trailing_silence_units=2, sample_rate=10))

    assert len(chunks) == 5
    assert [row[2] for row in chunks] == ["AUDIO", "AUDIO", "AUDIO_PADDED", "SILENCE", "SILENCE"]
    assert [row[3] for row in chunks] == [10, 10, 5, 0, 0]
    np.testing.assert_array_equal(chunks[2][1][:5], np.arange(20, 25, dtype=np.float32))
    np.testing.assert_array_equal(chunks[2][1][5:], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(chunks[3][1], np.zeros(10, dtype=np.float32))
