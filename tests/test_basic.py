import cmake_example as m
import numpy as np

def test_convert_to_arma():
    foo = np.array(np.linspace(3 * 4 * 5)).reshape(3, 4, 5)
    bar = np.zeros((3, 4, 5))
    np.testing.assert_allequal(m.add_cube(foo, bar), foo)
    np.testing.assert_allequal(m.add_mat(foo[0], bar[0]), foo[0])
    np.testing.assert_allequal(m.add_vec(foo[0, 0], bar[0, 0]), foo[0, 0])
