# DEVELOPMENT TOOL: separate from the frozen 17 October four-path publication protocol.
# See docs/BENCHMARK_TOOL_STATUS.md and benchmarks/notebooks/all-paths-speed.ipynb.
"""Run the corrected v0.2 equal-work speed benchmark.

Original benchmark records remain unchanged in Git history and benchmarks/raw.
This entry point no longer runs the non-equivalent legacy OLD wrapper.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from honest_gdn2 import main
if __name__=='__main__':raise SystemExit(main())
