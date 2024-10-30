import pytest
import numpy

from sos.system import System


def test_2_states():
    t_dips_2s = numpy.array([
        [[1., 0, 0], [1.5, 0, 0]],  # 0→x
        [[1.5, 0, 0], [2., 0, 0]],  # 1→x
    ])

    system = System([.5, ], t_dips_2s)

    w = .1

    assert (
        system.response_tensor_element_g((0, 0), [-w, w]) ==
        pytest.approx(system.response_tensor_element_f((0, 0), [-w, w]))
    )

    assert (
        system.response_tensor_element_g((0, 0, 0), [-2 * w, w, w]) ==
        pytest.approx(system.response_tensor_element_f((0, 0, 0), [-2 * w, w, w]))
    )


def test_3_states():
    t_dips_3s = numpy.array([
        [[1., 0, 0], [1.5, 0, 0], [.5, 0, 0]],  # 0→x
        [[1.5, 0, 0], [2., 0, 0], [.25, 0, 0]],  # 1→x
        [[.5, 0, 0], [.25, 0, 0], [1.5, 0, 0]]  # 2→x
    ])

    system = System([.5, .7], t_dips_3s)

    w = .1

    assert (
        system.response_tensor_element_g((0, 0), [-w, w]) ==
        pytest.approx(system.response_tensor_element_f((0, 0), [-w, w]))
    )

    assert (
        system.response_tensor_element_g((0, 0, 0), [-2 * w, w, w]) ==
        pytest.approx(system.response_tensor_element_f((0, 0, 0), [-2 * w, w, w]))
    )
