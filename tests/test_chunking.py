import numpy as np

from slackbench.dataset import full_second_chunks


def test_full_second_chunks_drops_tail():
    audio = np.arange(25, dtype=np.float32)
    chunks = list(full_second_chunks(audio, sample_rate=10))
    assert len(chunks) == 2
    assert chunks[0][0] == 0
    assert chunks[1][0] == 1
    assert len(chunks[0][1]) == 10
    assert len(chunks[1][1]) == 10
    assert chunks[1][1][-1] == 19
