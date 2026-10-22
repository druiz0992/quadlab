import numpy as np

from robotlab.sensors.dropout import GpsDropout


def test_disabled_and_total_random_loss():
    healthy = GpsDropout()
    failed = GpsDropout(probability=1.0)
    for stamp in range(10):
        assert not healthy.drop(stamp)
        assert failed.drop(stamp)
    assert failed.summary() == dict(
        received=10,
        delivered=0,
        random_drops=10,
        scheduled_drops=0,
        zone_drops=0,
        manual_drops=0,
    )


def test_outage_boundaries_and_combination_counts():
    model = GpsDropout(probability=1, outages_s=[[1, 2], [1.5, 3]])
    for stamp in (0, 999_999_999, 1_000_000_000, 2_000_000_000, 3_000_000_000):
        assert model.drop(stamp)
    assert model.summary() == dict(
        received=5,
        delivered=0,
        random_drops=3,
        scheduled_drops=2,
        zone_drops=0,
        manual_drops=0,
    )


def test_seeded_loss_repeats_without_modifying_global_rng():
    before = np.random.get_state()
    first = GpsDropout(probability=0.4, seed=37)
    second = GpsDropout(probability=0.4, seed=37)
    third = GpsDropout(probability=0.4, seed=38)
    decisions = [first.drop(t) for t in range(100)]
    assert decisions == [second.drop(t) for t in range(100)]
    assert decisions != [third.drop(t) for t in range(100)]
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_zone_blockout_drops_only_inside_and_takes_precedence_over_random():
    zone = {"min_m": [0, 0, 0], "max_m": [10, 10, 5]}
    model = GpsDropout(zones=[zone])
    assert model.drop(0, [5, 5, 1])
    assert model.drop(1, [10, 10, 5])  # boundary is inside (closed box)
    assert not model.drop(2, [11, 5, 1])
    assert not model.drop(3, None)  # unknown position cannot be in a zone
    assert model.summary()["zone_drops"] == 2
    lossy = GpsDropout(probability=1, zones=[zone])
    lossy.drop(0, [5, 5, 1])
    assert lossy.summary()["zone_drops"] == 1 and lossy.summary()["random_drops"] == 0
