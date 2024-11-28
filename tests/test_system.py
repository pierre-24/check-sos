import numpy
import io

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


def test_read_system():

    # 2-state
    system_2s_def = """1
    1 0.7
    0 0 1.0 0.0 0.0
    0 1 1.5 0.0 0.0
    1 1 2.0 0.0 0.0"""

    f = io.StringIO(system_2s_def)

    system = System.from_file(f)

    assert numpy.allclose(system.e_exci, [0, 0.7])
    assert numpy.allclose(system.t_dips, t_dips_2s)

    # 3-state
    system_3s_def = """2
    1 0.7
    2 0.9
    0 0 1. .5 0
    0 1 1.5 0 0
    0 2 .5 .5 0
    1 1 2. 0 .5
    1 2 .25 0 0
    2 2 1.5 0 .5"""

    f = io.StringIO(system_3s_def)

    system = System.from_file(f)

    assert numpy.allclose(system.e_exci, [0, 0.7, 0.9])
    assert numpy.allclose(system.t_dips, t_dips_3s)


def test_non_resonant_divergent():
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
            system_2s.response_tensor(fields, w, method=SOSMethod.FLUCT_DIVERGENT)
        )

        assert numpy.allclose(
            system_3s.response_tensor(fields, w, method=SOSMethod.GENERAL),
            system_3s.response_tensor(fields, w, method=SOSMethod.FLUCT_DIVERGENT)
        )


def test_non_resonant_non_divergent():
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
            system_2s.response_tensor(fields, w, method=SOSMethod.FLUCT_DIVERGENT),
            system_2s.response_tensor(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT)
        )

        assert numpy.allclose(
            system_3s.response_tensor(fields, w, method=SOSMethod.FLUCT_DIVERGENT),
            system_3s.response_tensor(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT)
        )


def test_non_divergent_not_harmonic_generation():
    """Check that non-divergent formula holds (i.e., nothing is imaginary) when the process involves a static field
    """

    system_3s = System([.7, .9], t_dips_3s)

    w = .1

    for fields in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, -1, 1)]:
        print(fields)

        t = system_3s.response_tensor(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT)
        assert all(x != numpy.inf for x in t.flatten())


def test_resonant_divergent():
    """
    Check that resonant and non-resonant formula provide the same result if damping is 0
    """

    system_2s = System([.7, ], t_dips_2s)
    system_3s = System([.7, .9], t_dips_3s)

    w = .1

    for n in range(1, 5):
        fields = tuple(1 for _ in range(n))
        print(fields)

        tr2s = system_2s.response_tensor_resonant(fields, w, method=SOSMethod.GENERAL)

        assert numpy.allclose(
            system_2s.response_tensor(fields, w, method=SOSMethod.GENERAL),
            tr2s.real
        )

        assert numpy.allclose(tr2s.imag, 0)

        tr3s = system_3s.response_tensor_resonant(fields, w, method=SOSMethod.GENERAL)

        assert numpy.allclose(
            system_3s.response_tensor(fields, w, method=SOSMethod.GENERAL),
            tr3s.real
        )

        assert numpy.allclose(tr3s.imag, 0)
