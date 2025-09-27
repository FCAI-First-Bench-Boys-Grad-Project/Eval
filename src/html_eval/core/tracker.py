# html_eval/core/tracker_base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import wandb
import mlflow

class BaseTracker(ABC):
    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Abstract base tracker.

        Args:
            experiment_name: Name of the experiment/run.
            config: Dictionary of hyperparameters/configs.
        """
        self.experiment_name = experiment_name
        self.config = config or {}

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_artifact(self, path: str, name: Optional[str] = None, type_: str = "artifact"):
        pass

    @abstractmethod
    def finish(self):
        pass


class WandbTracker(BaseTracker):
    def __init__(self, project: Optional[str] = None, **kwargs):
        super().__init__(kwargs.get("experiment_name"), kwargs.get("config"))

        self.project = project
        self.run = wandb.init(
            project=self.project,
            name=self.experiment_name,
            config=self.config,
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        wandb.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        wandb.config.update(params, allow_val_change=True)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: str = "artifact"):
        artifact = wandb.Artifact(name or path, type=type_)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self):
        wandb.finish()



class MLflowTracker(BaseTracker):
    def __init__(self, experiment_name: Optional[str] = None, run_name: Optional[str] = None, **kwargs):
        super().__init__(experiment_name, kwargs.get("config"))

        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        self.run = mlflow.start_run(run_name=run_name)

        # log initial config
        if self.config:
            mlflow.log_params(self.config)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_artifact(self, path: str, name: Optional[str] = None, type_: str = "artifact"):
        mlflow.log_artifact(path, artifact_path=name)

    def finish(self):
        mlflow.end_run()


def get_tracker(backend: str, **kwargs) -> BaseTracker:
    backend = backend.lower()
    if backend == "mlflow":
        return MLflowTracker(**kwargs)
    elif backend == "wandb":
        return WandbTracker(**kwargs)
    else:
        raise ValueError(f"Unknown tracker backend: {backend}")