import argparse
import sys
import itertools
import math

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
        print(' {sx}{x}{i}{sy}'.format(
            i=tr_[i],
            x='.' * (len(tensor.shape) - 1),
            sx=' ' * int(math.floor((13 - len(tensor.shape)) / 2)),
            sy=' ' * int(math.ceil((13 - len(tensor.shape)) / 2)),
        ), end='')
    print()

    for c in itertools.product(range(3), repeat=n - 1):
        print(''.join(tr_[x] for x in c), end=' ')
        for ci in range(3):
            print(' {: .6e}'.format(tensor[c][ci]), end='')
        print()


def print_tensor_resonant(tensor: NDArray):
    n = len(tensor.shape)

    print(' ' * n, end='')
    for i in range(3):
        print(' {sx}Re[{x}{i}]{sy} {sx}Im[{x}{i}]{sy}'.format(
            i=tr_[i],
            x='.' * (len(tensor.shape) - 1),
            sx=' ' * int(math.floor((14 - len(tensor.shape) - 5) / 2)),
            sy=' ' * int(math.ceil((14 - len(tensor.shape) - 5) / 2)),
        ), end='')
    print()

    for c in itertools.product(range(3), repeat=n - 1):
        print(''.join(tr_[x] for x in c), end=' ')
        for ci in range(3):
            print(' {: .6e} {: .6e}'.format(tensor[c][ci].real, tensor[c][ci].imag), end='')
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
    parser.add_argument('-d', '--damping', type=float, default=0, help='damping (in au)')

    args = parser.parse_args()

    print(get_preamble(args.fields, args.omega))

    system = System.from_file(args.input, args.nstates)

    if args.eV:
        system.e_exci /= AU_TO_EV

    if args.damping <= .0:
        print_tensor(system.response_tensor(input_fields=args.fields, frequency=args.omega))
    else:
        print_tensor_resonant(
            system.response_tensor_resonant(input_fields=args.fields, frequency=args.omega, damping=args.damping)
        )


if __name__ == '__main__':
    main()
