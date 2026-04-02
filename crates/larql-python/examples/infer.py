"""
Inference from a vindex — Rust attention + vindex walk FFN.

No MLX. No GPU. The entire forward pass runs in Rust with mmap'd weights.
Peak memory: one layer at a time. Model size: unlimited.

Usage:
    python examples/infer.py [path/to/model.vindex]
"""

import sys
import larql

vindex = larql.load(sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex")
print(vindex)
print()

prompts = [
    "The capital of France is",
    "Albert Einstein was a",
    "The programming language Python was created by",
    "Water boils at",
    "The largest planet in our solar system is",
]

for prompt in prompts:
    result = vindex.infer(prompt, top_k_predictions=3)
    top = result[0]
    others = ", ".join(f"{t} ({p:.1%})" for t, p in result[1:3])
    print(f"  {prompt}")
    print(f"    → {top[0]} ({top[1]:.1%})  also: {others}")
    print()
