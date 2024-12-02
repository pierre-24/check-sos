import pathlib
import itertools
import numpy
import io

import pytest

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
    {}""".format('\n'.join(
        '{} {} {:.3f} {:.3f} {:.3f}'.format(
            p[0], p[1], *t_dips_2s[*p]) for p in itertools.combinations_with_replacement(range(2), 2)
    ))

    f = io.StringIO(system_2s_def)

    system = System.from_file(f)

    assert numpy.allclose(system.e_exci, [0, 0.7])
    assert numpy.allclose(system.t_dips, t_dips_2s)

    # 3-state
    system_3s_def = """2
    1 0.7
    2 0.9
    {}""".format('\n'.join(
        '{} {} {:.3f} {:.3f} {:.3f}'.format(
            p[0], p[1], *t_dips_3s[*p]) for p in itertools.combinations_with_replacement(range(3), 2)
    ))

    f = io.StringIO(system_3s_def)

    system = System.from_file(f)

    assert numpy.allclose(system.e_exci, [0, 0.7, 0.9])
    assert numpy.allclose(system.t_dips, t_dips_3s)


@pytest.fixture
def system_2s():
    with (pathlib.Path(__file__).parent / '2-state.txt').open() as f:
        return System.from_file(f)


@pytest.fixture
def system_3s():
    with (pathlib.Path(__file__).parent / '3-state.txt').open() as f:
        return System.from_file(f)


def test_non_resonant_divergent(system_2s, system_3s):
    """
    Test the divergent cases (general formula vs fluctuation dipole with divergent formula for secular terms),
    so check against harmonic generation
    """

    w = .02

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


def test_non_resonant_non_divergent(system_2s, system_3s):
    """
    Test the non-divergent case (fluctuation dipole with divergent vs non-divergent formula for secular terms).
    """

    w = .02

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


def test_non_divergent_not_harmonic_generation(system_3s):
    """Check that non-divergent formula holds (i.e., nothing is infinite) when the process involves a static field
    """

    w = .02

    for fields in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, -1, 1)]:
        print(fields)

        t = system_3s.response_tensor(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT)
        assert all(x != numpy.inf for x in t.flatten())


def test_resonant_divergent_no_damping(system_2s, system_3s):
    """
    Check that resonant and non-resonant formula provide the same result if damping is 0
    """

    w = .02

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


def test_resonant_fluctuation_no_damping(system_2s, system_3s):
    """
    Check that general and fluctuation formula provide the same result if damping is 0
    """

    w = .02

    for n in range(1, 5):
        fields = tuple(1 for _ in range(n))
        print(fields)

        assert numpy.allclose(
            system_2s.response_tensor_resonant(fields, w, method=SOSMethod.GENERAL),
            system_2s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_DIVERGENT)
        )

        assert numpy.allclose(
            system_3s.response_tensor_resonant(fields, w, method=SOSMethod.GENERAL),
            system_3s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_DIVERGENT)
        )


def test_resonant_damping_2s_alpha(system_2s):
    """
    Check that general and fluctuation formula give the same result if damping is not 0, for alpha.
    """

    damping = 1e-2

    for w in [.1, .2, .3, .35]:

        assert numpy.allclose(
            system_2s.response_tensor_resonant((0, ), w, damping=damping, method=SOSMethod.GENERAL),
            system_2s.response_tensor_resonant((0, ), w, damping=damping, method=SOSMethod.FLUCT_DIVERGENT)
        )


def test_resonant_damping_2s_beta():
    """
    Check effect of damping on beta (where general and fluctuation formula do not give the same result!)
    against 2-state frequency dispersion formula from Berkovic et al. (https://doi.org/10.1063/1.480991).
    """

    # TODO: what if the ground state has an infinite lifetime, then?
    def f_berkovic(w0, w, g):
        g_ = g * 1j

        return w0 ** 2 / 3 * (1 / (
            (w0 - g_ - 2 * w) * (w0 - g_ - w)
        ) + 1 / (
            (w0 + g_ + w) * (w0 - g_ - w)
        ) + 1 / (
            (w0 + g_ + 2 * w) * (w0 + g_ + w)
        ))

    def beta_w_g(num, w0, w, g):
        # from the formula by berkovic

        g_ = g * 1j
        return 2 * num * (1 / (
            (w0 - g_ - 2 * w) * (w0 - g_ - w)
        ) + 1 / (
            (w0 + g_ + w) * (w0 - g_ - w)
        ) + 1 / (
            (w0 + g_ + 2 * w) * (w0 + g_ + w)
        ))

    def beta_w_g2(num, wx, wy, w, gx, gy):
        # a more complex version of the previous one

        gx_, gy_ = gx * 1j, gy * 1j
        return 2 * num * (1 / (
            (wx - gx_ - 2 * w) * (wy - gy_ - w)
        ) + 1 / (
            (wx + gx_ + w) * (wy - gy_ - w)
        ) + 1 / (
            (wx + gx_ + w) * (wy + gy_ + 2 * w)
        ))

    def beta_w_gx(num, w0, w, g):
        # separate real and imaginary part, if any
        g_ = g * 1j

        A = w0 + w
        B = w0 + 2 * w

        d = (A**2 + g**2) * (B**2 + g**2)
        real_part = (A * B - g**2) / d
        imag_part = -g_ * (w0 + w0 + 3 * w) / d

        A = w0 - 2 * w
        B = w0 - w

        d = (A**2 + g**2) * (B**2 + g**2)
        real_part += (A * B - g**2) / d
        imag_part += g_ * (w0 + w0 - 3 * w) / d

        A = w0 + w
        B = w0 - w

        d = (A**2 + g**2) * (B**2 + g**2)
        real_part += (A * B + g**2) / d
        imag_part += g_ * (2 * w) / d

        return 2 * num * (real_part + imag_part)

    w0 = .7
    system_2s = System([w0, ], t_dips_2s)
    num = system_2s.t_dips[0, 1, 0] ** 2 * (system_2s.t_dips[1, 1, 0] - system_2s.t_dips[0, 0, 0])

    damping = 1e-2

    # TODO: no damping on the static value?
    b0 = system_2s.response_tensor_element_f((0, 0, 0), [0, 0, 0])
    assert numpy.allclose(b0, beta_w_g(num, w0, 0, 0))

    for w in [.1, .2, .3, .35]:
        bwg = system_2s.response_tensor_element_g((0, 0, 0), [-2 * w, w, w], damping=damping)

        assert numpy.allclose(
            bwg,
            beta_w_g2(system_2s.t_dips[0, 0, 0] ** 3, 0, 0, w, damping, damping)
            + beta_w_g2(system_2s.t_dips[1, 0, 0] ** 2 * system_2s.t_dips[0, 0, 0], w0, 0, w, damping, damping)  # noqa
            + beta_w_g2(system_2s.t_dips[1, 0, 0] ** 2 * system_2s.t_dips[0, 0, 0], 0, w0, w, damping, damping)  # noqa
            + beta_w_g2(system_2s.t_dips[1, 0, 0] ** 2 * system_2s.t_dips[1, 1, 0], w0, w0, w, damping, damping) # noqa
        )

        bwgf = system_2s.response_tensor_element_f((0, 0, 0), [-2 * w, w, w], damping=damping)

        """Not equal, but close:
        print(
            bwgf,
            beta_w_g2(system_2s.t_dips[0, 0, 0] ** 3, 0, 0, w, 0, 0)
            + beta_w_g2(system_2s.t_dips[1, 0, 0] ** 2 * system_2s.t_dips[0, 0, 0], w0, 0, w, damping, 0)  # noqa
            + beta_w_g2(system_2s.t_dips[1, 0, 0] ** 2 * system_2s.t_dips[0, 0, 0], 0, w0, w, 0, damping)  # noqa
            + beta_w_g2(system_2s.t_dips[1, 0, 0] ** 2 * system_2s.t_dips[1, 1, 0], w0, w0, w, damping, damping)  # noqa
        )"""

        assert numpy.allclose(beta_w_g(num, w0, w, damping), bwgf)
        assert numpy.allclose(beta_w_g2(num, w0, w0, w, damping, damping), bwgf)
        assert numpy.allclose(beta_w_gx(num, w0, w, damping), bwgf)
        assert numpy.allclose(bwgf / b0, f_berkovic(w0, w, damping))

        assert not numpy.allclose(bwg, bwgf)


def test_resonant_fluctuation_damping(system_2s, system_3s):
    """
    Check that both fluctuation formula (with and without divergent secular terms) provide the same result
    """

    w = .02
    damping = 1e-1

    for n in range(1, 5):
        fields = tuple(1 for _ in range(n))
        print(fields)

        assert numpy.allclose(
            system_2s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_DIVERGENT, damping=damping),
            system_2s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT, damping=damping)
        )

        assert numpy.allclose(
            system_3s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_DIVERGENT, damping=damping),
            system_3s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT, damping=damping)
        )


def test_resonant_non_divergent_not_harmonic_generation(system_3s):
    """Check that in the case of resonant formula, using divergent and non-divergent actually provide the same result.

    TODO: ... But should that be the case?
    """

    w = .02
    damping = 1e-1

    for fields in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, -1, 1)]:
        print(fields)

        assert numpy.allclose(
            system_3s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_DIVERGENT, damping=damping),
            system_3s.response_tensor_resonant(fields, w, method=SOSMethod.FLUCT_NON_DIVERGENT, damping=damping)
        )
