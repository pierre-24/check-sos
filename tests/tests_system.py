import numpy

from sos.system import System, SOSMethod

t_dips_2s = numpy.array([
    [[1., 0, 0], [1.5, 0, 0]],  # 0→x
    [[1.5, 0, 0], [2., 0, 0]],  # 1→x
])

t_dips_3s = numpy.array([
    [[1., .5, 0], [1.5, 0, 0], [.5, .5, 0]],  # 0→x
    [[1.5, 0, 0], [2., 0, .5], [.25, 0, 0]],  # 1→x
    [[.5, .5, 0], [.25, 0, 0], [1.5, 0, .5]]  # 2→x
])


def test_divergent():
    """
    Test the divergent cases (general formula vs fluctuation dipole with divergent formula for secular terms),
    so check against harmonic generation
    """

    system_2s = System([.7, ], t_dips_2s)
    system_3s = System([.7, .9], t_dips_3s)

    w = .1

    for n in range(1, 6):
        fields = tuple(1 for _ in range(n))
        print(fields)

        assert numpy.allclose(
            system_2s.response_tensor(fields, w, method=SOSMethod.GENERAL),
            system_2s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_DIVERGENT)
        )

        assert numpy.allclose(
            system_3s.response_tensor(fields, w, method=SOSMethod.GENERAL),
            system_3s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_DIVERGENT)
        )


def test_non_divergent():
    """
    Test the non-divergent case (fluctuation dipole with divergent vs non-divergent formula for secular terms).
    """

    system_2s = System([.7, ], t_dips_2s)
    system_3s = System([.7, .9], t_dips_3s)

    w = .1

    for n in range(1, 5):
        fields = tuple(1 for _ in range(n))
        print(fields)

        assert numpy.allclose(
            system_2s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_DIVERGENT),
            system_2s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_NONDIVERGENT)
        )

        assert numpy.allclose(
            system_3s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_DIVERGENT),
            system_3s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_NONDIVERGENT)
        )


def test_non_divergent_not_harmonic_generation():
    """Check that non-divergent formula holds (i.e., nothing is imaginary) when the process involves a static field
    """

    system_3s = System([.7, .9], t_dips_3s)

    w = .1

    for i in range(4):
        fields = tuple(1 if j < i else 0 for j in range(3))
        print(fields)

        t = system_3s.response_tensor(fields, w, method=SOSMethod.FLUCTUATION_NONDIVERGENT)
        assert all(x != numpy.inf for x in t.flatten())
