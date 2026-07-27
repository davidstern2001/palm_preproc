#!/usr/bin/env python3
"""palm_preproc entry point.

Usage:
    python run_preproc.py -c config/<project>.yaml [-n] [-v|-q] [--force]
                          [--stages domains report clip merge] [--log-datetime]
"""
import sys
from palm_preproc.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
