from cli import parse_args

def main():
    args = parse_args()

    # Here you would typically load the pipeline class dynamically
    # For example:
    # module = __import__(args.pipeline_path.stem)
    # pipeline_class = getattr(module, args.pipeline_class)

    # Then you would instantiate the pipeline and run the evaluation
    # pipeline = pipeline_class(args)
    # results = pipeline.run_evaluation()

    print(f"Running evaluation with dataset: {args.dataset}")
    print(f"Using pipeline class: {args.pipeline_class}")
    print(f"Output directory: {args.output_dir}")
    print(f"Limit: {args.limit}, Seed: {args.seed}, Batch size: {args.batch_size}")


if __name__ == "__main__":
    main()
