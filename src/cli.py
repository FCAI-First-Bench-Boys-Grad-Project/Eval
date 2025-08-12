from typing import Optional
from pathlib import Path
import argparse
from dataclasses import dataclass

@dataclass
class CLIArgs:
    dataset: str
    pipeline_path: Path
    pipeline_class: str
    output_dir: Optional[Path]
    limit: Optional[int]
    seed: int
    batch_size: int

def parse_args() -> CLIArgs:
    parser = argparse.ArgumentParser(description="Parse evaluation arguments")

    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. swde or websrc") # Maybe add all?
    parser.add_argument("--pipeline-path", type=Path, required=True, help="Path to pipeline Python file")
    parser.add_argument("--pipeline-class", default="Pipeline", help="Pipeline class name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save results") # Might not need it if we are going to use mlflow
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")

    args = parser.parse_args()

    return CLIArgs(
        dataset=args.dataset,
        pipeline_path=args.pipeline_path,
        pipeline_class=args.pipeline_class,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
        batch_size=args.batch_size,
    )
