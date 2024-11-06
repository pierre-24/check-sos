import numpy
import pytest

from few_state.system_maker import make_system_from_E2tT, make_system_from_mtT

CT_dipole_2s = lambda mu_CT: [numpy.array([.0, .0, mu_CT])]

# Eq. (A3)
CT_dipole_3s = lambda mu_CT, theta: [
    mu_CT * numpy.array([numpy.sin(theta), .0, numpy.cos(theta)]),
    mu_CT * numpy.array([-numpy.sin(theta), .0, numpy.cos(theta)]),
]

# Eq. (A4)
CT_dipole_4s = lambda mu_CT, theta: [
    mu_CT * numpy.array([.0, numpy.sin(theta), numpy.cos(theta)]),
    mu_CT / 2 * numpy.array([
        numpy.sqrt(3) * numpy.sin(theta),
        -numpy.sin(theta),
        2 * numpy.cos(theta)
    ]),
    mu_CT / 2 * numpy.array([
        -numpy.sqrt(3) * numpy.sin(theta),
        -numpy.sin(theta),
        2 * numpy.cos(theta)
    ]),
]


def _get_energies(E_VB, E_CT, t, T, n):

    V = E_CT - E_VB

    E_0 = .5 * (E_VB + E_CT - (n - 1) * T - numpy.sqrt((V - (n - 1) * T) ** 2 + 4 * n * t ** 2))
    E_e1 = E_CT + T
    E_f = .5 * (E_VB + E_CT - (n - 1) * T + numpy.sqrt((V - (n - 1) * T) ** 2 + 4 * n * t ** 2))

    return E_0, E_e1, E_f


def test_make_system_E2tT():
    E_VB, E_CT = -1, 1
    t = .5
    T = .1
    theta = numpy.deg2rad(45)
    mu_CT = 1

    # 2-state system
    n = 1
    system_2s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_2s(theta))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_2s.e_exci[0] == .0
    assert system_2s.e_exci[1] == pytest.approx(E_f - E_0)

    # 3-state system
    n = 2
    system_3s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_3s(mu_CT, theta))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_3s.e_exci[0] == .0
    assert system_3s.e_exci[1] == pytest.approx(E_e1 - E_0)
    assert system_3s.e_exci[2] == pytest.approx(E_f - E_0)

    # 4-state system
    n = 3
    system_4s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_4s(mu_CT, theta))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_4s.e_exci[0] == .0
    assert system_4s.e_exci[1] == pytest.approx(E_e1 - E_0)
    assert system_4s.e_exci[2] == pytest.approx(E_e1 - E_0)
    assert system_4s.e_exci[3] == pytest.approx(E_f - E_0)


def test_make_system_mtT_dipoles():
    """Check dipoles. See Chapter 11 of my thesis for more details
    """

    m_CT = .1
    t = .05
    T = .01
    theta = numpy.deg2rad(60)
    mu_CT = 1.

    # 2-state system
    system_2s = make_system_from_mtT(m_CT, t, T, CT_dipole_2s(mu_CT))

    assert system_2s.t_dips[0, 0][2] == pytest.approx(mu_CT * (1 + m_CT) / 2)
    assert system_2s.t_dips[0, 1][2] == pytest.approx(-mu_CT / 2 * numpy.sqrt(1 - m_CT ** 2))
    assert system_2s.t_dips[1, 1][2] == pytest.approx(mu_CT * (1 - m_CT) / 2)

    # 3-state system
    system_3s = make_system_from_mtT(m_CT, t, T, CT_dipole_3s(mu_CT, theta))

    assert numpy.allclose(system_3s.t_dips[0, 0], [.0, .0, mu_CT * (1 + m_CT) / 2 * numpy.cos(theta)])
    assert numpy.allclose(system_3s.t_dips[0, 1], [-mu_CT * numpy.sqrt((1 + m_CT) / 2) * numpy.sin(theta), .0, .0])
    assert numpy.allclose(system_3s.t_dips[0, 2], [.0, .0, -mu_CT / 2 * numpy.sqrt(1 - m_CT ** 2) * numpy.cos(theta)])
    assert numpy.allclose(system_3s.t_dips[1, 1], [.0, .0, mu_CT * numpy.cos(theta)])
    assert numpy.allclose(system_3s.t_dips[1, 2], [mu_CT * numpy.sqrt((1 - m_CT) / 2) * numpy.sin(theta), .0, .0])
    assert numpy.allclose(system_3s.t_dips[2, 2], [.0, .0, mu_CT * (1 - m_CT) / 2 * numpy.cos(theta)])

    # 4-state system
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    assert numpy.allclose(system_4s.t_dips[0, 0], [.0, .0, mu_CT * (1 + m_CT) / 2 * numpy.cos(theta)])

    assert numpy.allclose(
        system_4s.t_dips[0, 1],
        mu_CT / 4 * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [numpy.sqrt(2) * numpy.sin(theta), -numpy.sqrt(6) * numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(
        system_4s.t_dips[0, 2],
        mu_CT / 4 * numpy.sqrt(2) * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [-numpy.sqrt(3) * numpy.sin(theta), -numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(
        system_4s.t_dips[0, 3],
        mu_CT / 2 * numpy.sqrt(1 - m_CT**2) * numpy.array([0, 0, -numpy.cos(theta)])
    )

    assert numpy.allclose(
        system_4s.t_dips[1, 1],
        mu_CT / 4 * numpy.array([numpy.sqrt(3) * numpy.sin(theta), numpy.sin(theta), 4 * numpy.cos(theta)])
    )

    assert numpy.allclose(
        system_4s.t_dips[1, 2],
        mu_CT / 4 * numpy.array([-numpy.sin(theta), numpy.sqrt(3) * numpy.sin(theta), 0])
    )

    assert numpy.allclose(
        system_4s.t_dips[1, 3],
        mu_CT / 4 * numpy.sqrt(2) * numpy.sqrt((1 - m_CT) / 2) * numpy.array(
            [-numpy.sin(theta), numpy.sqrt(3) * numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(
        system_4s.t_dips[2, 2],
        mu_CT / 4 * numpy.array([-numpy.sqrt(3) * numpy.sin(theta), -numpy.sin(theta), 4 * numpy.cos(theta)])
    )

    assert numpy.allclose(
        system_4s.t_dips[2, 3],
        mu_CT / 4 * numpy.sqrt(2) * numpy.sqrt((1 - m_CT) / 2) * numpy.array(
            [numpy.sqrt(3) * numpy.sin(theta), numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(system_4s.t_dips[3, 3], [.0, .0, mu_CT * (1 - m_CT) / 2 * numpy.cos(theta)])


def test_make_system_mtT_beta():
    m_CT = -.2
    t = .05
    T = .01
    theta = numpy.deg2rad(60)
    mu_CT = 1.

    # 2-state system (Eq. A2)
    system_2s = make_system_from_mtT(m_CT, t, T, CT_dipole_2s(mu_CT))

    t_2s = system_2s.response_tensor((1, 1), frequency=0)

    assert t_2s[2, 2, 2] == pytest.approx(-3. / 8 * m_CT * (1 - m_CT**2)**2 * mu_CT**3 / t**2)
    assert all(x == .0 for x in t_2s.flatten()[:-1])

    # 3-state system (Eq. A4)
    system_3s = make_system_from_mtT(m_CT, t, T, CT_dipole_3s(mu_CT, theta))

    t_3s = system_3s.response_tensor((1, 1), frequency=0)

    assert t_3s[2, 2, 2] == pytest.approx(-3. / 16 * m_CT * (1 - m_CT**2)**2 * mu_CT**3 / t**2 * numpy.cos(theta) ** 3)

    assert t_3s[2, 0, 0] == pytest.approx((1 - m_CT**2) / 2 * mu_CT**3 * numpy.sin(theta) ** 2 * numpy.cos(theta) * (
        (2 * T + t * numpy.sqrt(2 * (1 - m_CT) / (1 + m_CT)))**-2 + 2 * (
            2 * t * numpy.sqrt(2 / (1 - m_CT**2)) * (2 * T + t * numpy.sqrt(2 * (1 - m_CT) / (1 + m_CT)))
        )**-1
    ))

    # 4-state system
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)

    assert t_4s[2, 2, 2] == pytest.approx(-1. / 8 * m_CT * (1 - m_CT**2)**2 * mu_CT**3 / t**2 * numpy.cos(theta) ** 3)

    assert t_4s[2, 1, 1] == pytest.approx((1 - m_CT**2) / 4 * mu_CT**3 * numpy.sin(theta) ** 2 * numpy.cos(theta) * (
        (3 * T + t * numpy.sqrt(3 * (1 - m_CT) / (1 + m_CT)))**-2 + 2 * (
            2 * t * numpy.sqrt(3 / (1 - m_CT**2)) * (3 * T + t * numpy.sqrt(3 * (1 - m_CT) / (1 + m_CT)))
        )**-1
    ))

    assert t_4s[1, 1, 1] == pytest.approx(3 * (1 + m_CT) / 4 * mu_CT**3 * numpy.sin(theta) ** 3 * (
        (3 * T + t * numpy.sqrt(3 * (1 - m_CT) / (1 + m_CT)))**-2
    ))
