# -*- coding: utf-8 -*-
"""Run every revised supplementary figure script in numerical order."""

from pathlib import Path
import subprocess
import sys

import figure_common as F


def main() -> None:
    here = Path(__file__).resolve().parent
    scripts = sorted(here.glob("[0-9][0-9]_plot_*.py"))
    for script in scripts:
        print(f"Running {script.name}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=here, check=True)
    F.make_manifest()
    print(f"Completed {len(scripts)} figure scripts: {here}")


if __name__ == "__main__":
    main()

