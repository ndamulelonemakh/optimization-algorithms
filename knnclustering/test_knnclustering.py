import pytest
import numpy as np

from knnclustering import eucleadian_distance


@pytest.mark.parametrize('a,b,expected',
                         [(5, 1, 4.0), (0, 1, 1.0)])
def test_1d_eucleadian_distance(a, b, expected):
    calculated_distance = eucleadian_distance(a, b)
    assert calculated_distance == expected


@pytest.mark.parametrize('a,b,expected',
                         [
                             ([3, 2], [4, 1], np.sqrt(2)),
                             ([1, 2], [3, 5], np.sqrt(13))
                         ])
def test_2d_eucleadian_distance(a, b, expected):
    calculated_distance = eucleadian_distance(a, b)
    assert np.isclose(calculated_distance, expected)


if __name__ == '__main__':
    pytest.main()
