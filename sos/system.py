import collections
import itertools
import math
import enum
import more_itertools
import numpy

from typing import List, Iterable
from numpy.typing import NDArray


class ComponentsIterator:
    """Iterate over (unique) components of a NLO tensor
    """

    def __init__(self, input_fields: Iterable[int]):
        self.fields = [-sum(input_fields)] + list(input_fields)

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
    GENERAL = enum.auto
    FLUCTUATION = enum.auto


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

    def response_tensor(
            self, input_fields: tuple = (1, 1), frequency: float = 0, method: SOSMethod = SOSMethod.GENERAL) -> NDArray:
        """Get a response tensor, a given SOS formula
        """

        compute_component = {
            SOSMethod.GENERAL: self.response_tensor_element_g,
            SOSMethod.FLUCTUATION: self.response_tensor_element_f
        }[method]

        it = ComponentsIterator(input_fields)
        t = numpy.zeros(numpy.repeat(3, len(it.fields)))
        e_fields = list(i * frequency for i in it.fields)

        for c in it:
            component = compute_component(c, e_fields)

            for ce in it.reverse(c):
                t[ce] = component

        return t

    def response_tensor_element_g(self, component: tuple, e_fields: List[float]) -> float:
        """Compute the value of a component of a response tensor, using the most generic formula
        """

        print('g')

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
                print(p, states, dips)

                ens = [
                    self.e_exci[e] + sum(p[j][1] for j in range(i + 1)) for i, e in enumerate(states)
                ]

                print(ens)

                value += numpy.prod(dips) / numpy.prod(ens)

        return value * num_perm

    def response_tensor_element_f(self, component: tuple, e_fields: List[float]) -> float:
        """Compute the value of a component of a response tensor, using fluctuation dipoles.
        Does not work for `len(component) > 3`.
        """

        print('f')

        assert len(component) == len(e_fields)

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

                print(p, states, dips)

                ens = [
                    self.e_exci[e] + sum(p[j][1] for j in range(i + 1)) for i, e in enumerate(states)
                ]

                print(ens)

                value += numpy.prod(dips) / numpy.prod(ens)

        return value * num_perm
