from celery import Celery
from src.eval.html_datasets.base import BaseHTMLDataset
from src.eval import RerankerPipeline, Experiment, Evaluator, WebSrcDataset

# Example for all possible experiements
datasets = [WebSrcDataset(), BaseHTMLDataset()]
pipelines = [RerankerPipeline()]
evaluators = [Evaluator(["token_f1"])]

list_of_inputs = pipelines * datasets * evaluators
experiments = [Experiment(*inputs[0]) for inputs in list_of_inputs]


# Example for single experiment
single_experiment = Experiment(WebSrcDataset(), pipeline=RerankerPipeline,
                        evaluator=Evaluator(["token_f1"]))

experiments.append(single_experiment)

for experiment in experiments:
    # notice that run's inputs are passed to delay instead
    experiment.run.delay(batch_size= None, shuffle=False)