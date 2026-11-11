"""sales-performance file for ensuring the package is executable
as `sales-performance` and `python -m sales_performance`
"""

import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs) -> Any:
    """Ejecuta el comando ``kedro run`` del proyecto desde la línea de comandos."""
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    interactive = hasattr(sys, "ps1")
    kwargs["standalone_mode"] = not interactive

    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
