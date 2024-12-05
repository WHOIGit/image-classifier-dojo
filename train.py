import os, pathlib

import src.train
from src.train import argparse_init, args_subsetter_factory, argparse_runtime_args, main

if __name__ == '__main__':
    import sys
    PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
    if sys.path[0] != str(PROJECT_ROOT): sys.path.insert(0, str(PROJECT_ROOT))
    parser = argparse_init()
    args_subsetter = args_subsetter_factory(parser)
    def just_hparams(args_namespace):
        subparsers = ['Model Parameters', 'Epoch Parameters']
        return args_subsetter(args_namespace, [], subparsers)
    src.train.just_hparams = just_hparams
    args = parser.parse_args()
    argparse_runtime_args(args)
    main(args)


