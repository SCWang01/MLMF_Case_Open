"""Render manuscript Figs. 3-6 using the archive root's figure_data.json."""
from pathlib import Path
import sys

from figure_code.render_case_figures import main


if __name__ == "__main__":
    source = Path(__file__).with_name("figure_data.json")
    sys.argv[1:1] = ["--data", str(source)]
    main()
