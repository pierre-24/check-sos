import numpy

from sos.system import System

from typing import List
from numpy.typing import NDArray


def _get_dipole_matrix(CT_dipoles: List[NDArray], C: NDArray) -> NDArray:
    """Get the (MO) dipole matrix

    :param C: coefficient matrix, `(ao, MO)`
    :param CT_dipoles: list of CT dipoles
    """
    n = len(CT_dipoles)

    AO_dipole_matrix = numpy.zeros((n + 1, n + 1, 3))
    for i in range(1, n + 1):
        AO_dipole_matrix[i, i] = CT_dipoles[i - 1]

    """Legacy:
    MO_dipoles = numpy.zeros((n + 1, n + 1, 3))
    for i in range(0, n + 1):
        for j in range(0, n + 1):
            for a in range(0, n + 1):
                MO_dipoles[i, j] += AO_dipole_matrix[a, a] * C[i, a] * C[j, a]
    return MO_dipoles
    """

    return numpy.einsum(
        'ilm,jl->ijm', numpy.einsum('ij,jlm->ilm', C, AO_dipole_matrix), C)


def make_system_from_hamiltonian(hamiltonian: NDArray, CT_dipoles: List[NDArray]) -> System:
    n = len(CT_dipoles)
    assert hamiltonian.shape == (n + 1, n + 1)

    eigv, eigh = numpy.linalg.eigh(hamiltonian)

    e_exci = eigv[1:] - eigv[0]

    C = eigh.T  # state is the first index

    # dipoles
    dipole_matrix = _get_dipole_matrix(CT_dipoles, C)

    return System(list(e_exci), dipole_matrix)


def make_system_from_E2tT(E_VB: float, E_CT: float, t: float, T: float, CT_dipoles: List[NDArray]) -> System:
    n = len(CT_dipoles)

    # create hamiltonian
    H = numpy.zeros((n + 1, n + 1))

    H[0, 0] = E_VB
    for i in range(1, n + 1):
        H[0, i] = -t
        H[i, 0] = -t

        for j in range(1, n + 1):
            if i == j:
                H[i, i] = E_CT
            else:
                H[i, j] = H[j, i] = -T

    # the rest is handled by the general function
    return make_system_from_hamiltonian(H, CT_dipoles)


def make_system_from_mtT(m_CT: float, t: float, T: float, CT_dipoles: List[NDArray]) -> System:
    n = len(CT_dipoles)

    # energies
    E_ge = n * T + t * numpy.sqrt(n * (1 - m_CT) / (1 + m_CT))
    E_gf = 2 * t * numpy.sqrt(n / (1 - m_CT ** 2))

    e_exci = [E_ge] * (n - 1) + [E_gf]

    # coefficients
    C = numpy.zeros((n + 1, n + 1))

    C[0, 0] = numpy.sqrt((1 - m_CT) / 2)
    C[0, 1:] = numpy.sqrt((1 + m_CT) / (2 * n))  # phi_g

    C[-1, 0] = numpy.sqrt((1 + m_CT) / 2)
    C[-1, 1:] = -numpy.sqrt((1 - m_CT) / (2 * n))  # phi_f

    # phi_ei
    for i in range(1, n):
        C[i, i + 1] = i / numpy.sqrt(i * (i + 1))
        C[i, 1:i + 1] = -1 / numpy.sqrt(i * (i + 1))

    # dipoles
    dipole_matrix = _get_dipole_matrix(CT_dipoles, C)

    return System(list(e_exci), dipole_matrix)
