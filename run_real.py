"""Real-Model Experiment Runner: Per-layer W4A4 rotation on actual LLMs.

Usage:
    python run_real.py --model mistral-7b
    python run_real.py --model llama-3-8b
"""

import sys
sys.path.insert(0, ".")

from src.experiments.run_real_experiments import main

if __name__ == "__main__":
    main()
