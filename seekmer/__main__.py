#!/usr/bin/env python3

import argparse
import pathlib
import sys

import logbook
import logbook.more
import seekmer


def _add_index_subparser(subparsers):
    parser = subparsers.add_parser(
        'index',
        help='build K-mer index for target sequences',
    )
    parser.add_argument(
        'fasta',
        help='specify transcriptome FASTA file',
        type=pathlib.Path,
    )
    parser.add_argument(
        'index',
        type=pathlib.Path,
        help='specify output index file',
    )
    return subparsers


def _add_infer_subparser(subparsers):
    parser = subparsers.add_parser(
        'infer',
        help='infer isoform expression and gene fusion',
    )
    parser.add_argument(
        'index',
        help='specify output index file',
        type=pathlib.Path,
    )
    parser.add_argument(
        'fastq1',
        help='specify the first FASTQ file',
        type=pathlib.Path,
    )
    parser.add_argument(
        'fastq2',
        help='specify the second FASTQ file',
        type=pathlib.Path,
    )
    parser.add_argument(
        'report',
        help='specify the output report file',
        type=pathlib.Path,
    )
    return subparsers


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=('K-mer based RNA-seq isoform quantification and fusion '
                     'detection'),
    )
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='seekmer 0.0.1',
    )
    subparsers = parser.add_subparsers(
        dest='subcommand',
        metavar='SUBCOMMAND',
    )
    subparsers = _add_index_subparser(subparsers)
    subparsers = _add_infer_subparser(subparsers)
    return vars(parser.parse_args())


def main():
    log_handler = logbook.more.ColorizedStderrHandler()
    log_handler.format_string = ('{record.time:%Y-%m-%d %H:%M:%S} '
                                 '[{record.level_name}] {record.channel}: '
                                 '{record.message}')
    log_handler.push_application()
    args = _parse_arguments()
    if args['subcommand'] == 'index':
        return seekmer.index(**args)
    elif args['subcommand'] == 'infer':
        return seekmer.infer(**args)


if __name__ == '__main__':
    sys.exit(main())
