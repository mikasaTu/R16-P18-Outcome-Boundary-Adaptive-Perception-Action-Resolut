#!/usr/bin/env python3
"""Entry point for preregistered stopping diagnostics.

It intentionally shares the frozen new evaluator implementation while retaining
a distinct source binding.  Formal invocations must pass one of the four modes
and `--record-trace`.
"""

from evaluate_checkpoint_grid import main


if __name__ == "__main__":
    main()
