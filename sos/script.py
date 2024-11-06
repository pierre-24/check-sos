import argparse
import sys
import itertools

from sos import HC_IN_EV, AU_TO_EV
from sos.system import System

from numpy.typing import NDArray


tr_ = {0: 'x', 1: 'y', 2: 'z'}


def get_preamble(fields, w) -> str:

    def _fx(f) -> str:
        if f == 0:
            return '0'
        elif f == 1:
            return 'w'
        elif f == -1:
            return '-w'
        else:
            return '{}w'.format(f)

    return 'X({};{}) @ w={}'.format(
        _fx(-sum(fields)),
        ','.join(_fx(f) for f in fields),
        '{:.1f}nm'.format(HC_IN_EV / w) if w != .0 else '0'
    )


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
    parser.add_argument('-i', '--input', help='System definition', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('-e', '--eV', help='energies are in eV', action='store_true')
    parser.add_argument('-n', '--nstates', type=int, help='number of excited states, if not provided', default=-1)
    parser.add_argument('-f', '--fields', type=get_fields, default='1 1', help='List of input fields')
    parser.add_argument('-w', '--omega', type=float, default=0, help='laser frequency (in au)')

    args = parser.parse_args()

    print(get_preamble(args.fields, args.omega))

    system = System.from_file(args.input, args.nstates)

    if args.eV:
        system.e_exci /= AU_TO_EV

    print_tensor(system.response_tensor(input_fields=args.fields, frequency=args.omega))


if __name__ == '__main__':
    main()
