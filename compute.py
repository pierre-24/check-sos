import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='source', type=argparse.FileType('r'), default=sys.stdin)

    args = parser.parse_args()


if __name__ == '__main__':
    main()
