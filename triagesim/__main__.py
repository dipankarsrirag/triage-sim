import sys

from triagesim.cli import main

if __name__ == "__main__":  # vLLM may spawn worker processes that re-import __main__
    sys.exit(main())
