import argparse
import sys
import numpy

from few_state.system_maker import make_system_from_mtT
from sos import AU_TO_EV


def get_mct(inp: str) -> float:
    try:
        m_CT = float(inp)
    except ValueError:
        raise argparse.ArgumentTypeError('m_CT should be float')

    if m_CT < -1 or m_CT > 1:
        raise argparse.ArgumentTypeError('m_CT should be between -1 and 1')

    return m_CT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help='List of CT dipoles', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('-e', '--eV', help='output energies in eV instead of au', action='store_true')
    parser.add_argument('-t', '--transfer-VB-CT', type=float, help='transfer integral', default=.05)
    parser.add_argument('-T', '--transfer-CT-CT', type=float, help='transfer integral', default=.01)
    parser.add_argument('-m', '--mix', type=get_mct, default=.1, help='mixing parameter')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout)

    args = parser.parse_args()

    # extract CT dipoles
    CT_dipoles = []
    for i, line in enumerate(args.source.readlines()):

        # empty line
        if line.strip() == '':
            continue

        # comment
        if line[0] == '#':
            continue

        try:
            dipole = numpy.array([float(x) for x in line.split()])
        except ValueError:
            raise Exception('dipole on line {} should contains only floats'.format(i + 1))

        if len(dipole) != 3:
            raise Exception('dipole on line {} should have 3 components'.format(i + 1))

        CT_dipoles.append(dipole)

    # create system
    system = make_system_from_mtT(args.mix, args.transfer_VB_CT, args.transfer_CT_CT, CT_dipoles)

    if args.eV:
        system.e_exci *= AU_TO_EV

    system.to_file(args.output)


if __name__ == '__main__':
    main()
