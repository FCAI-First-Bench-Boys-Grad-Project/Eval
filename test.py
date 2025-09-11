# test.py
import os

# -------------------------
# Set environment early (before importing anything that may init vllm / HF / CUDA)
# -------------------------
os.environ['NVIDIA_API_KEY'] = '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'
os.environ["VLLM_USE_V1"] = "0"
# os.environ["VLLM_LOG_LEVEL"] = "ERROR"   # reduce vLLM INFO spam

# Now import your package (safe: env vars are already set)
from eval import Experiment
from eval import WebSrcDataset, SWDEDataset
from eval import RerankerPipeline
from eval import Evaluator


def main():
    data = SWDEDataset(
        local_dir="/home/abdo/PAPER/Eval/data/swde/hf_SWDE",
        indices=list(range(0, 200, 2))
    )

    data.set_domain("camera")

    pipeline = RerankerPipeline()

    evaluator = Evaluator(
        evaluation_metrics=data.evaluation_metrics,
    )

    exp = Experiment(
        data=data,
        pipeline=pipeline,
        evaluator=evaluator,
    )

    pred, gt, res = exp.run(batch_size=1)

    print("Results:", res)
    print('=' * 50)
    print("Predictions:", pred)
    print('=' * 50)
    print("Ground Truth:", gt)
    print('=' * 50)
    # If evaluator.metrics exists and you want to inspect a particular metric:
    if evaluator.metrics:
        print("Scores: ", evaluator.metrics[0].scores)


if __name__ == "__main__":
    # Only set start method in the top-level script (not inside imported modules).
    # Use try/except because it raises if already set.
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    main()
