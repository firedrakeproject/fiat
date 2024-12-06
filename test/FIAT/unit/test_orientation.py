import pytest
from FIAT.reference_element import Point, UFCInterval, UFCTriangle, UFCQuadrilateral
from FIAT.orientation_utils import make_entity_permutations_tensorproduct


def test_orientation_make_entity_permutations_tensorproduct():
    cells = [UFCInterval(), UFCInterval()]
    m = make_entity_permutations_tensorproduct(cells, [1, 0], [{0: [0, 1],
                                                                1: [1, 0]},
                                                               {0: [0]}])
    assert m == {(0, 0, 0): [0, 1],
                 (0, 1, 0): [1, 0]}
    m = make_entity_permutations_tensorproduct(cells, [1, 1], [{0: [0, 1],
                                                                1: [1, 0]},
                                                               {0: [0, 1],
                                                                1: [1, 0]}])
    assert m == {(0, 0, 0): [0, 1, 2, 3],
                 (0, 0, 1): [1, 0, 3, 2],
                 (0, 1, 0): [2, 3, 0, 1],
                 (0, 1, 1): [3, 2, 1, 0],
                 (1, 0, 0): [0, 2, 1, 3],
                 (1, 0, 1): [2, 0, 3, 1],
                 (1, 1, 0): [1, 3, 0, 2],
                 (1, 1, 1): [3, 1, 2, 0]}


@pytest.mark.parametrize("cell", [Point(), UFCInterval(), UFCTriangle(), UFCQuadrilateral()])
def test_orientation_cell_orientation_reflection_map(cell):
    # Check cell reflections relative to orientation 0 or (0, 0, 0).
    m = cell.cell_orientation_reflection_map()
    if isinstance(cell, Point):
        assert m == {0: 0}
    elif isinstance(cell, UFCInterval):
        # o     0        1
        #
        #     0---1    1---0
        #
        assert m == {0: 0,
                     1: 1}
    elif isinstance(cell, UFCTriangle):
        # o     0      1      2      3      4      5
        #
        #      2      1      2      0      1      0
        #      | \    | \    | \    | \    | \    | \
        #      0--1   0--2   1--0   1--2   2--0   2--1
        #
        assert m == {0: 0,
                     1: 1,
                     2: 1,
                     3: 0,
                     4: 0,
                     5: 1}
    elif isinstance(cell, UFCQuadrilateral):
        # eo\\io    0      1      2      3
        #
        #        1---3  0---2  3---1  2---0
        #   0    |   |  |   |  |   |  |   |
        #        0---2  1---3  2---0  3---1
        #
        #        2---3  3---2  0---1  1---0
        #   1    |   |  |   |  |   |  |   |
        #        0---1  1---0  2---3  3---2
        #
        assert m == {(0, 0, 0): 0,
                     (0, 0, 1): 1,
                     (0, 1, 0): 1,
                     (0, 1, 1): 0,
                     (1, 0, 0): 1,
                     (1, 0, 1): 0,
                     (1, 1, 0): 0,
                     (1, 1, 1): 1}
