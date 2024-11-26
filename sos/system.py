import collections
import itertools
import math
import enum
import more_itertools
import numpy

from typing import List, Iterable, Self, TextIO
from numpy.typing import NDArray


class ComponentsIterator:
    """Iterate over (unique) components of a NLO tensor
    """

    def __init__(self, input_fields: Iterable[int], use_full: bool = True):
        assert all(type(x) is int for x in input_fields)

        self.fields = [-sum(input_fields)] + list(input_fields)

        # small trick to differentiate \omega_\sigma from the rest if intrinsic
        if not use_full:
            self.fields[0] = 's'

        self.each = collections.Counter(self.fields)
        self.last = {}

        # prepare a ideal→actual map
        N = 0
        self.ideal_to_actual = {}

        for c, n in self.each.items():
            self.last[c] = N
            N += n

        for i, field in enumerate(self.fields):
            self.ideal_to_actual[i] = self.last[field]
            self.last[field] += 1

        self.actual_to_ideal = dict((b, a) for a, b in self.ideal_to_actual.items())

    def __len__(self) -> int:
        """Return the number of components that the iterator will yield, which is a combination of
        combination with replacements for each type of field
        (see https://www.calculatorsoup.com/calculators/discretemathematics/combinationsreplacement.php)
        """
        return int(numpy.prod([math.factorial(3 + n - 1) / math.factorial(n) / 2 for n in self.each.values()]))

    def __iter__(self) -> Iterable[tuple]:
        yield from self.iter()

    def _iter_reordered(self, perms) -> Iterable[tuple]:
        """Perform the cartesian product of all permutation, then reorder so that it matches `self.fields`
        """

        for ideal_component in itertools.product(*perms.values()):
            ideal_component = list(more_itertools.collapse(ideal_component))
            yield tuple(ideal_component[self.ideal_to_actual[i]] for i in range(len(self.fields)))

    def iter(self) -> Iterable[tuple]:
        """Iterate over unique components.
        It is a cartesian product between combination for each type of field, then reordered.
        """

        # perform combinations for each "type" of field
        perms = {}
        for c, n in self.each.items():
            perms[c] = list(itertools.combinations_with_replacement(range(3), n))

        # cartesian product
        yield from self._iter_reordered(perms)

    def reverse(self, component: tuple) -> Iterable[tuple]:
        """Yield all possible (unique) combination of a given component"""

        assert len(component) == len(self.fields)

        # fetch components for each type of fields
        c_with_field = dict((f, []) for f in self.each.keys())
        for i, field in enumerate(self.fields):
            c_with_field[field].append(component[i])

        # permute
        perms = {}
        for field, components in c_with_field.items():
            perms[field] = list(more_itertools.unique_everseen(itertools.permutations(components)))

        # cartesian product
        yield from self._iter_reordered(perms)


class SOSMethod(enum.Enum):
    GENERAL = enum.auto()
    FLUCT_DIVERGENT = enum.auto()
    FLUCT_NON_DIVERGENT = enum.auto()


class System:
    """A system, represented by a list of excitation energies and corresponding transition dipoles
    """

    def __init__(self, e_exci: List[float], t_dips: NDArray):

        assert len(e_exci) == t_dips.shape[0] - 1

        e_exci.insert(0, 0)

        self.e_exci = numpy.array(e_exci)
        self.t_dips = t_dips

    def __len__(self):
        return len(self.e_exci)

    @classmethod
    def from_file(cls, f: TextIO, n: int = -1) -> Self:
        """Read out a file.

        :param f: an opened file
        :param n: number of excitation energies
        """

        lines = f.readlines()

        # if number of state is not provided, it is in the first line
        if n < 0:
            n = int(lines.pop(0).strip())

        if n < 1:
            raise Exception('Expect n > 0, got {}'.format(n))

        # read excitations energies
        e_exci = [.0] * n

        if len(lines) < n:
            raise Exception('Not enough lines in the files, at least {} excitations energies are required'.format(n))

        for i, line in enumerate(lines[:n]):
            chunks = line.split()
            if len(chunks) != 2:
                raise Exception('Incorrect number of value for excitation energies at line {}'.format(i + 1))
            try:
                iexci, eexci = int(chunks[0]), float(chunks[1])
            except ValueError:
                raise Exception('Invalid excitation energy at line {}'.format(i + 1))

            if iexci > n or iexci < 1:
                raise Exception('Incorrect energy att excitation {} at line {}'.format(iexci, i + 1))

            e_exci[iexci - 1] = eexci

        # read transition dipoles
        t_dips = numpy.zeros((n + 1, n + 1, 3))

        for i, line in enumerate(lines[n:]):
            chunks = line.split()
            if len(chunks) != 5:
                raise Exception('Incorrect number of value for transition dipole at line {}'.format(n + i + 2))

            iexci, jexci, x, y, z = int(chunks[0]), int(chunks[1]), float(chunks[2]), float(chunks[3]), float(chunks[4])

            if iexci > n or iexci < 0 or jexci > n or jexci < 0:
                raise Exception('Incorrect transition {}→{} at line {}'.format(iexci, jexci, n + i + 2))

            t_dips[iexci, jexci] = t_dips[jexci, iexci] = [x, y, z]

        return cls(e_exci, t_dips)

    def to_file(self, f: TextIO):
        f.write('{}\n'.format(len(self) - 1))
        for i, e in enumerate(self.e_exci[1:]):
            f.write('{} {:.7f}\n'.format(i + 1, e))

        for i in range(len(self)):
            for j in range(0, i + 1):
                f.write('{} {} {:.7f} {:.7f} {:.7f}\n'.format(i, j, *self.t_dips[i, j]))

    def response_tensor(
            self,
            input_fields: tuple = (1, 1),
            frequency: float = 0,
            method: SOSMethod = SOSMethod.FLUCT_NON_DIVERGENT,
    ) -> NDArray:
        """Get a response tensor, a given SOS formula
        """

        if len(input_fields) == 0:
            raise Exception('input fields is empty?!?')

        compute_component = {
            SOSMethod.GENERAL: self.response_tensor_element_nr_g,
            SOSMethod.FLUCT_DIVERGENT: self.response_tensor_element_nr_f,
            SOSMethod.FLUCT_NON_DIVERGENT: lambda c_, e_: self.response_tensor_element_nr_f(c_, e_, False)
        }[method]

        it = ComponentsIterator(input_fields)
        t = numpy.zeros(numpy.repeat(3, len(it.fields)))
        e_fields = list(i * frequency for i in it.fields)

        for c in it:
            component = compute_component(c, e_fields)

            for ce in it.reverse(c):
                t[ce] = component

        return t

    def response_tensor_element_nr_g(self, component: tuple, e_fields: List[float]) -> float:
        """Compute the value of a component of a response tensor, using the most generic formula, Eq. (1) of text.
        """

        assert len(component) == len(e_fields)

        value = .0

        to_permute = list(zip(component, e_fields))
        num_perm = numpy.prod([math.factorial(i) for i in collections.Counter(to_permute).values()])

        for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
            for states in itertools.product(range(0, len(self)), repeat=len(component) - 1):
                stx = list(states)
                stx.append(0)
                stx.insert(0, 0)

                dips = [self.t_dips[stx[i], stx[i + 1], p[i][0]] for i in range(len(component))]

                ens = [
                    self.e_exci[e] + sum(p[j][1] for j in range(i + 1)) for i, e in enumerate(states)
                ]

                value += numpy.prod(dips) / numpy.prod(ens)

        return value * num_perm

    def response_tensor_element_nr_f(
            self, component: tuple, e_fields: List[float], use_divergent: bool = True) -> float:
        """
        Compute the value of a component of a response tensor, using fluctuation dipoles.
        It corresponds to Eq. (6) of the text.

        Note: it breaks for n > 6, since `(ab)_a1(cd)_a3(efg)_a5a6` appears and that I did not yet implement a
        general scheme. The n=5 case is handled using an ad-hoc correction ;)
        """

        assert len(component) == len(e_fields)
        assert len(e_fields) < 7

        value = .0

        to_permute = list(zip(component, e_fields))
        num_perm = numpy.prod([math.factorial(i) for i in collections.Counter(to_permute).values()])

        for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
            for states in itertools.product(range(1, len(self)), repeat=len(component) - 1):
                stx = list(states)
                stx.append(0)
                stx.insert(0, 0)

                dips = [
                    self.t_dips[stx[i], stx[i + 1], p[i][0]] - (
                        0 if stx[i] != stx[i + 1]
                        else self.t_dips[0, 0, p[i][0]]
                    ) for i in range(len(component))
                ]

                ens = [
                    self.e_exci[e] + sum(p[j][1] for j in range(i + 1)) for i, e in enumerate(states)
                ]

                value += numpy.prod(dips) / numpy.prod(ens)

        if len(component) > 3:
            for set_g in range(1, len(component) - 2):
                if use_divergent:
                    value += self._secular_term_divergent(component, e_fields, (set_g, ))
                else:
                    value += self._secular_term_non_divergent(component, e_fields, set_g)

        # ad hoc correction for n=5
        if len(component) == 6 and use_divergent:
            value += self._secular_term_divergent(component, e_fields, (1, 3))

        return value * num_perm

    def _secular_term_divergent(self, component: tuple, e_fields: List[float], set_ground: tuple) -> float:
        """Compute the additional secular term that happen when n > 2, but using a divergent definition.

        Implements parts of the Eq. (8) of the text, by setting in Eq. (7) the term a_i for all i in `set_ground`
        to the ground state.
        """

        value = .0
        to_permute = list(zip(component, e_fields))

        for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
            for states in itertools.product(range(1, len(self)), repeat=len(component) - 1 - len(set_ground)):
                states = list(states)

                for g in set_ground:
                    states.insert(g, 0)

                stx = list(states)
                stx.append(0)
                stx.insert(0, 0)

                dips = [
                    self.t_dips[stx[i], stx[i + 1], p[i][0]] - (
                        0 if stx[i] != stx[i + 1]
                        else self.t_dips[0, 0, p[i][0]]
                    ) for i in range(len(component))
                ]

                ens = [
                    self.e_exci[e] + sum(p[j][1] for j in range(i + 1)) for i, e in enumerate(states)
                ]

                value += numpy.prod(dips) / numpy.prod(ens)

        return value

    def _secular_term_non_divergent(self, component: tuple, e_fields: List[float], set_ground: int) -> float:
        """Implement Eq. (9) to provide a non-divergent secular term.
        """

        value = .0

        to_permute = list(zip(component, e_fields))

        for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
            x = -sum(p[j][1] for j in range(set_ground + 1))

            for states in itertools.product(range(1, len(self)), repeat=len(component) - 2):
                states = list(states)

                states.insert(set_ground, 0)

                stx = list(states)
                stx.append(0)
                stx.insert(0, 0)

                # numerator of Eq. (9)
                dips = [
                    self.t_dips[stx[i], stx[i + 1], p[i][0]] - (
                        0 if stx[i] != stx[i + 1]
                        else self.t_dips[0, 0, p[i][0]]
                    ) for i in range(len(component))
                ]

                # denominator of Eq. (9)
                # TODO: there is probably a way to write a nicer code here, without all those `continue`.
                for i in range(len(component) - 1):
                    if i == set_ground:
                        continue

                    ens = []

                    for l_ in range(i + 1):
                        if l_ == set_ground:
                            continue

                        ens.append(self.e_exci[states[l_]] + sum(p[j][1] for j in range(l_ + 1)))

                    for l_ in range(i, len(component) - 1):
                        if l_ == set_ground:
                            continue

                        ens.append(self.e_exci[states[l_]] + sum(p[j][1] for j in range(l_ + 1)) + x)

                    value += numpy.prod(dips) / numpy.prod(ens)

        return -.5 * value
