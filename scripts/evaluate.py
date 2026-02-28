#!/usr/bin/env python3
"""Run the evaluation pipeline against the golden dataset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.runner import main

if __name__ == "__main__":
    main()
