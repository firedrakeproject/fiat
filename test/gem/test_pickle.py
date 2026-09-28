import pickle

import pytest

import gem


@pytest.mark.parametrize(
    "name,extent",
    [
        (None, None),
        ("A", None),
        ("A", 5),
        (None, 5),
    ],
)
def test_pickle_index(name, extent):
    idx = gem.Index(name=name, extent=extent)
    new_idx = pickle.loads(pickle.dumps(idx))

    assert idx.name == new_idx.name
    assert idx.extent == new_idx.extent
    # assert idx.count == new_idx.count
