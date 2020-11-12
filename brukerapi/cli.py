from argparse import ArgumentParser
from brukerapi.splitters import *
from brukerapi.folders import *
import sys


def main():
    """

    """
    parser = ArgumentParser(prog='bruker')
    subparsers = parser.add_subparsers()

    # report sub-command
    parser_report = subparsers.add_parser('report', help='export properties of data sets to json, or yaml file')
    parser_report.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to a folder containing Bruker data",
    )
    parser_report.add_argument(
        "-o",
        "--output",
        help="Path to a folder containing Bruker data",
    )
    parser_report.add_argument(
        "-f",
        "--format",
        choices=['json', 'yml'],
        default='json',
        help="Format of report files",
    )
    parser_report.add_argument(
        "-p",
        "--props",
        type=str,
        nargs='+',
        help="List of properties to include",
    )
    parser_report.add_argument(
        "-v",
        "--verbose",
        help="make verbose",
        action="store_true"
    )
    parser_report.set_defaults(func=report)

    # report sub-command
    parser_split = subparsers.add_parser('split', help='split dataset into several sub-datasets')
    parser_split.add_argument(
        "-i",
        "--input",
        dest="path_in",
        type=str,
        required=True,
        help="Bruker 2dseq data set",
    )
    parser_split.add_argument(
        "-o",
        "--output",
        dest="path_out",
        type=str,
        required=False,
        help="Folder to save splitted data sets",
    )
    parser_split.add_argument(
        "-s",
        "--slice_package",
        dest="slice_package",
        action='store_true',
        help="Split by slice package",
    )
    parser_split.add_argument(
        "-f",
        "--frame_group",
        dest="frame_group",
        type=str,
        help="Split by frame group",
    )
    parser_split.set_defaults(func=split)

    # filter sub-command
    parser_filter = subparsers.add_parser('filter', help='get files based on querry')
    parser_filter.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to a folder containing Bruker data",
    )
    parser_filter.add_argument(
        "-q",
        "--query",
        type=str,
        required=True,
        help="Filter query",
    )
    parser_filter.set_defaults(func=filter)

    args = parser.parse_args()
    args.func(args)


def split(args):
    """
    split sub-command
    :param args:
    :return:
    """

    dataset = Dataset(args.path_in)

    if args.slice_package:
        SlicePackageSplitter().split(dataset, write=True)
    elif args.frame_group:
        FrameGroupSplitter(args.frame_group).split(dataset, write=True)


def report(args):
    """
    report sub - command
    :param args:
    :return:
    """
    input = Path(args.input)

    if args.output is None:
        output = None
    else:
        output = Path(args.output)

    if input.is_dir():
        # folder in-place
        if output is None:
            Folder(input).report(format_=args.format, props=args.props, verbose=args.verbose)
        elif output.is_dir():
            # folder to folder
            Folder(input).report(path_out=output, format_=args.format, props=args.props, verbose=args.verbose)
    else:
        # dataset in-place
        if output is None:
            Dataset(input, add_parameters=['subject']).report(props=args.props, verbose=args.verbose)
        # dataset to folder, or dataset to file
        elif output.is_dir():
            Dataset(input, add_parameters=['subject']).report(path=output, props=args.props, verbose=args.verbose)


def filter(args):
    folder = Folder(args.input)
    Filter(args.query, recursive=True, in_place=True).filter(folder)

    # print to std out
    for dataset in folder.dataset_list_rec:
        print(str(dataset.path), file=sys.stdout)


if __name__ == "__main__":
    main()
