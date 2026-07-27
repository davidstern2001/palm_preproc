"""Allow `python -m palm_preproc` as an alternative to run_preproc.py."""
import sys
from .pipeline import main

if __name__ == "__main__":
    sys.exit(main())
