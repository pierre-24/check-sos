import collections
import itertools
import math
from typing import Tuple, List, Iterable

import more_itertools
import numpy
from numpy.typing import NDArray


class ComponentIterator:
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

    def __len__(self):
        return len(self.fields)

    def __iter__(self) -> Iterable[tuple]:
        """Iterate over unique components.
        """

        yield from self.iter()

    def iter(self) -> Iterable[tuple]:
        # perform permutations for each "type" of field
        perms = {}
        for c, n in self.each.items():
            perms[c] = list(itertools.combinations_with_replacement(range(3), n))

        # get component by performing a cartesian product, then reordering it
        for ideal_component in itertools.product(*perms.values()):
            ideal_component = list(more_itertools.collapse(ideal_component))
            yield tuple(ideal_component[self.ideal_to_actual[i]] for i in range(len(self)))

    def reverse(self, component: tuple) -> Iterable[tuple]:
        pass


class System:
    def __init__(self, energies: List[float], dipoles: NDArray):
        self.e_exci = energies
        self.t_dips = dipoles

    def response_tensor_g(self, input_fields: Tuple[int] = (1, 1), frequency: float = 0) -> NDArray:
        """Get a response tensor, using the most generic SOS formula

        :param input_fields: input fields
        :param frequency: input frequency
        """

        t = numpy.zeros(numpy.repeat(3, len(input_fields)))
        it = ComponentIterator(input_fields)
        e_fields = list(i * frequency for i in it.fields)

        for c in it:
            to_permute = list(zip(c, e_fields))
            num_perm = numpy.prod(math.factorial(i) for i in collections.Counter(to_permute).values())
            component = .0

            for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
                for states in itertools.product(range(1, self.N), repeat=len(input_fields)):
                    stx = list(states)
                    stx.append(0)
                    stx.insert(0, 0)

                    fnum = numpy.prod(self.dipoles[stx[i], stx[i + 1], p[i][0]] - (
                        0 if stx[i] != stx[i + 1]
                        else self.dipoles[0, 0, p[i][0]])
                        for i in range(len(input_fields) + 1))

                    fden = numpy.prod(self.energies[e] - self.energies[0] - p[i][1] for i, e in enumerate(states))

                    component += fnum / fden

            for ce in it.reverse(c):
                t[ce] = component * num_perm

        return t
