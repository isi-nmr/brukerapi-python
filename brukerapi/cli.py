from argparse import ArgumentParser
from brukerapi.splitters import *


def split():

    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="path_in",
        type=str,
        required=True,
        help="Bruker study folder.",
    )

    parser.add_argument(
        "-o",
        "--output",
        dest="path_out",
        type=str,
        required=False,
        help="Bruker study folder.",
    )

    parser.add_argument(
        "-sp",
        "--slice_package",
        dest="slice_package",
        action='store_true',
        help="Split by slice package",
    )

    parser.add_argument(
        "-fg",
        "--frame_group",
        dest="frame_group",
        type=str,
        help="Split by frame group",
    )

    args = parser.parse_args()

    dataset = Dataset(args.path_in)

    if args.slice_package:
        SlicePackageSplitter().split(dataset, write=True)
    elif args.frame_group:
        FrameGroupSplitter(args.frame_group).split(dataset, write=True)


if __name__ == "__main__":
    split()
