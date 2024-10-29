import itertools

import more_itertools

from sos.system import ComponentIterator


def test_iter_linear():
    # static
    assert set(ComponentIterator((0, )).iter()) == set(itertools.combinations_with_replacement(range(3), 2))

    # dynamic
    assert set(ComponentIterator((1, )).iter()) == set(itertools.product(range(3), range(3)))


def test_iter_quadratic():
    # static
    it_static = ComponentIterator((0, 0))
    assert it_static.fields == [0, 0, 0]
    assert set(it_static.iter()) == set(itertools.combinations_with_replacement(range(3), 3))

    # pockels (-w,w,0)
    it_pockels = ComponentIterator((1, 0))
    assert it_pockels.fields == [-1, 1, 0]
    assert set(it_pockels.iter()) == set(itertools.product(range(3), range(3), range(3)))

    # SHG (-2w;w,w)
    it_SHG = ComponentIterator((1, 1))
    assert it_SHG.fields == [-2, 1, 1]
    assert set(it_SHG.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), itertools.combinations_with_replacement(range(3), 2)
        )
    )


def test_iter_cubic():
    # static
    it_static = ComponentIterator((0, 0, 0))
    assert it_static.fields == [0, 0, 0, 0]
    assert set(it_static.iter()) == set(itertools.combinations_with_replacement(range(3), 4))

    # Kerr (-w,w,0, 0)
    it_kerr = ComponentIterator((1, 0, 0))
    assert it_kerr.fields == [-1, 1, 0, 0]
    assert set(it_kerr.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), range(3), itertools.combinations_with_replacement(range(3), 2)
        )
    )

    # DFWM (-w;w,-w,w)
    it_DFWM = ComponentIterator((1, -1, 1))
    assert it_DFWM.fields == [-1, 1, -1, 1]

    assert set(it_DFWM.iter()) == set(  # this one needs to be reordered on the fly
        (x[0], x[2], x[1], x[3]) for x in (
            tuple(more_itertools.collapse(i)) for i in itertools.product(
                itertools.combinations_with_replacement(range(3), 2),
                itertools.combinations_with_replacement(range(3), 2)
            )
        )
    )

    # EFISHG (-2w;w,w,0)
    it_EFISHG = ComponentIterator((1, 1, 0))
    assert it_EFISHG.fields == [-2, 1, 1, 0]
    assert set(it_EFISHG.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), itertools.combinations_with_replacement(range(3), 2), range(3)
        )
    )

    # THG (-3w,w,w,w)
    it_THG = ComponentIterator((1, 1, 1))
    assert it_THG.fields == [-3, 1, 1, 1]
    assert set(it_THG.iter()) == set(
        tuple(more_itertools.collapse(i)) for i in itertools.product(
            range(3), itertools.combinations_with_replacement(range(3), 3)
        )
    )
