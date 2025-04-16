import itertools
import math

import more_itertools

from sos.system import ComponentsIterator


def test_iter_linear():
    # static
    assert set(ComponentsIterator((0,)).iter()) == set(itertools.combinations_with_replacement(range(3), 2))

    # dynamic
    assert set(ComponentsIterator((1,)).iter()) == set(itertools.product(range(3), range(3)))


def test_iter_quadratic():
    # static
    it_static = ComponentsIterator((0, 0))
    assert it_static.fields == [0, 0, 0]
    assert len(it_static) == math.factorial(5) / (math.factorial(3) * 2)
    assert set(it_static.iter()) == set(itertools.combinations_with_replacement(range(3), 3))

    assert set(it_static.reverse((0, 0, 0))) == {(0, 0, 0)}
    assert set(it_static.reverse((0, 0, 1))) == set(itertools.permutations([0, 0, 1]))
    assert set(it_static.reverse((0, 1, 2))) == set(itertools.permutations([0, 1, 2]))

    # pockels (-w,w,0)
    it_pockels = ComponentsIterator((1, 0))
    assert it_pockels.fields == [-1, 1, 0]
    assert len(it_pockels) == 3 * 3 * 3
    assert set(it_pockels.iter()) == set(itertools.product(range(3), range(3), range(3)))

    assert set(it_pockels.reverse((0, 1, 2))) == {(0, 1, 2)}

    # SHG (-2w;w,w)
    it_SHG = ComponentsIterator((1, 1))
    assert it_SHG.fields == [-2, 1, 1]
    assert len(it_SHG) == 3 * math.factorial(3)
    assert set(it_SHG.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), itertools.combinations_with_replacement(range(3), 2)
        )
    )
    assert set(it_SHG.reverse((0, 0, 0))) == {(0, 0, 0)}

    assert set(it_SHG.reverse((0, 0, 1))) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            [0], itertools.permutations([0, 1])
        )
    )

    assert set(it_SHG.reverse((0, 1, 2))) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            [0], itertools.permutations([1, 2])
        )
    )


def test_iter_cubic():
    # static
    it_static = ComponentsIterator((0, 0, 0))
    assert it_static.fields == [0, 0, 0, 0]
    assert len(it_static) == math.factorial(6) / (math.factorial(4) * math.factorial(2))
    assert set(it_static.iter()) == set(itertools.combinations_with_replacement(range(3), 4))

    assert set(it_static.reverse((0, 0, 0, 0))) == {(0, 0, 0, 0)}
    assert set(it_static.reverse((0, 0, 0, 1))) == set(itertools.permutations([0, 0, 0, 1]))
    assert set(it_static.reverse((0, 1, 2, 1))) == set(itertools.permutations([0, 1, 2, 1]))

    # Kerr (-w,w,0, 0)
    it_kerr = ComponentsIterator((1, 0, 0))
    assert it_kerr.fields == [-1, 1, 0, 0]
    assert len(it_kerr) == 3 * 3 * math.factorial(3)
    assert set(it_kerr.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), range(3), itertools.combinations_with_replacement(range(3), 2)
        )
    )

    # DFWM (-w;w,-w,w)
    it_DFWM = ComponentsIterator((1, -1, 1))
    assert it_DFWM.fields == [-1, 1, -1, 1]
    assert len(it_DFWM) == math.factorial(3) * math.factorial(3)

    assert set(it_DFWM.iter()) == set(  # this one needs to be reordered on the fly
        (x[0], x[2], x[1], x[3]) for x in (
            tuple(more_itertools.collapse(i)) for i in itertools.product(
                itertools.combinations_with_replacement(range(3), 2),
                itertools.combinations_with_replacement(range(3), 2)
            )
        )
    )

    # EFISHG (-2w;w,w,0)
    it_EFISHG = ComponentsIterator((1, 1, 0))
    assert it_EFISHG.fields == [-2, 1, 1, 0]
    assert len(it_EFISHG) == 3 * 3 * math.factorial(3)
    assert set(it_EFISHG.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), itertools.combinations_with_replacement(range(3), 2), range(3)
        )
    )

    assert set(it_EFISHG.reverse((0, 1, 2, 1))) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            [0], itertools.permutations([1, 2]), [1]
        )
    )

    # THG (-3w,w,w,w)
    it_THG = ComponentsIterator((1, 1, 1))
    assert it_THG.fields == [-3, 1, 1, 1]
    assert len(it_THG) == 3 * math.factorial(5) / (math.factorial(3) * 2)
    assert set(it_THG.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), itertools.combinations_with_replacement(range(3), 3)
        )
    )

    assert set(it_THG.reverse((0, 1, 2, 1))) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            [0], itertools.permutations([1, 2, 1])
        )
    )
