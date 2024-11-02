import argparse
import sys
import itertools

from sos.system import System

from numpy.typing import NDArray


tr_ = {0: 'x', 1: 'y', 2: 'z'}

HC = 45.56335


def print_tensor(tensor: NDArray):
    n = len(tensor.shape)

    print(' ' * n, end='')
    for i in range(3):
        print(' {:^13}'.format(tr_[i]), end='')
    print()

    for c in itertools.product(range(3), repeat=n - 1):
        print(''.join(tr_[x] for x in c), end=' ')
        for ci in range(3):
            print(' {: .6e}'.format(tensor[c][ci]), end='')
        print()


def get_fields(inp: str) -> tuple:
    try:
        return tuple(int(x) for x in inp.split())
    except ValueError:
        raise argparse.ArgumentTypeError('invalid input for fields, expect only integers')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='System definition', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('-e', '--eV', help='energies are in eV', action='store_true')
    parser.add_argument('-n', '--nstates', type=int, help='number of excited states, if not provided', default=-1)
    parser.add_argument('-f', '--fields', type=get_fields, default='1 1', help='List of input fields')
    parser.add_argument('-w', '--omega', type=float, default=0, help='laser frequency (in au)')

    args = parser.parse_args()

    print('X({}) @ w={}'.format(
        ','.join('{}w'.format(f) for f in ([-sum(args.fields)] + list(args.fields))),
        '{:.1f}nm'.format(HC / args.omega) if args.omega != .0 else '0'
    ))

    try:
        system = System.from_file(args.source, args.eV, args.nstates)
        print_tensor(system.response_tensor(input_fields=args.fields, frequency=args.omega))
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    main()
