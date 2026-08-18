from slackbench.metrics import summarize


def test_summarize_basic():
    stats = summarize([1.0, 2.0, 3.0])
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["median"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_summarize_empty():
    stats = summarize([])
    assert stats["count"] == 0
    assert stats["mean"] is None
