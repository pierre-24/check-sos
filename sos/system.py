import collections
import itertools
import math
from typing import Tuple, List, Iterable

import more_itertools
import numpy
from numpy.typing import NDArray


class ComponentsIterator:
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


class System:
    def __init__(self, energies: List[float], dipoles: NDArray):
        energies.insert(0, 0)

        assert len(energies) == dipoles.shape[0]

        self.e_exci = numpy.array(energies)
        self.t_dips = dipoles

    def __len__(self):
        return len(self.e_exci)

    def response_tensor_g(self, input_fields: Tuple[int] = (1, 1), frequency: float = 0) -> NDArray:
        """Get a response tensor, using the most generic SOS formula

        :param input_fields: input fields
        :param frequency: input frequency
        """

        t = numpy.zeros(numpy.repeat(3, len(input_fields)))
        it = ComponentsIterator(input_fields)
        e_fields = list(i * frequency for i in it.fields)

        for c in it:
            to_permute = list(zip(c, e_fields))
            num_perm = numpy.prod([math.factorial(i) for i in collections.Counter(to_permute).values()])
            component = .0

            for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
                for states in itertools.product(range(1, len(self)), repeat=len(input_fields)):
                    stx = list(states)
                    stx.append(0)
                    stx.insert(0, 0)

                    fnum = numpy.prod([
                        self.t_dips[stx[i], stx[i + 1], p[i][0]] - (
                            0 if stx[i] != stx[i + 1]
                            else self.t_dips[0, 0, p[i][0]]
                        ) for i in range(len(input_fields) + 1)
                    ])

                    fden = numpy.prod([self.e_exci[e] - self.e_exci[0] - p[i][1] for i, e in enumerate(states)])

                    component += fnum / fden

            for ce in it.reverse(c):
                t[ce] = component * num_perm

        return t
