import pytest
import numpy

from sos.system import System


def test_divergent():
    """
    Test the divergent cases (general formula vs fluctuation with divergent formula for secular terms),
    so check against harmonic generation
    """

    t_dips_2s = numpy.array([
        [[1., 0, 0], [1.5, 0, 0]],  # 0→x
        [[1.5, 0, 0], [2., 0, 0]],  # 1→x
    ])

    system_2s = System([.7, ], t_dips_2s)

    t_dips_3s = numpy.array([
        [[1., 0, 0], [1.5, 0, 0], [.5, 0, 0]],  # 0→x
        [[1.5, 0, 0], [2., 0, 0], [.25, 0, 0]],  # 1→x
        [[.5, 0, 0], [.25, 0, 0], [1.5, 0, 0]]  # 2→x
    ])

    system_3s = System([.7, .9], t_dips_3s)

    w = .1

    for n in range(1, 6):
        component = tuple(0 for _ in range(n + 1))
        e_fields = [w for _ in range(n)]
        e_fields.insert(0, -n * w)

        print(e_fields)

        assert (
            system_2s.response_tensor_element_g(component, e_fields) ==
            pytest.approx(system_2s.response_tensor_element_f(component, e_fields, use_divergent=True))
        )

        assert (
            system_3s.response_tensor_element_g(component, e_fields) ==
            pytest.approx(system_3s.response_tensor_element_f(component, e_fields, use_divergent=True))
        )
