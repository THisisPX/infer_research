"""Phase 1.3 Pilot Experiment Runner.

Quick start:
    # Synthetic weights only
    python run.py --exp ALL --dist channel_outlier

    # All distributions
    python run.py --exp ALL --all-dists
"""

import sys
sys.path.insert(0, ".")

from src.experiments.run_experiments import main

if __name__ == "__main__":
    main()
