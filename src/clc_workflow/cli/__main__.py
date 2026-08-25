"""So `python -m clc_workflow.cli` works when the console script is not on PATH."""
import sys

from clc_workflow.cli.main import main

sys.exit(main())
