
## S3ArtifactStorage ##
import boto3, botocore

from aim.storage.artifacts import registry
from aim.storage.artifacts.s3_storage import S3ArtifactStorage

def S3ArtifactStorageFactory(**boto3_client_kwargs:dict):
    class S3ArtifactStorageCustom(S3ArtifactStorage):
        def _get_s3_client(self):
            if 'config' in boto3_client_kwargs and isinstance(boto3_client_kwargs['config'],dict):
                config_kwargs = boto3_client_kwargs.pop('config')
                boto3_client_kwargs['config'] = botocore.config.Config(**config_kwargs)
            client = boto3.client('s3', **boto3_client_kwargs)
            return client
    return S3ArtifactStorageCustom

def S3ArtifactStoragePatcher(**boto3_client_kwargs):
    registry.registry['s3'] = S3ArtifactStorageFactory(**boto3_client_kwargs)



## Lightning AimLogger ##
import importlib.util
from typing import Optional, Dict, Any

from aim.pytorch_lightning import AimLogger
from aim.sdk.adapters.pytorch_lightning import rank_zero_only
from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT

class AimLoggerWithContext(AimLogger):
    def __init__(
        self,
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        context_prefixes: Optional[Dict] = dict(subset={'train':'train_', 'val':'val_', 'test':'test_'}),
        context_postfixes: Optional[Dict] = dict(),
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        run_name: Optional[str] = None,
        run_hash: Optional[str] = None,
    ):
        super().__init__()
        self._experiment_name = experiment
        self._run_name = run_name
        self._repo_path = repo

        self._context_prefixes = context_prefixes
        self._context_postfixes = context_postfixes
        self._system_tracking_interval = system_tracking_interval
        self._log_system_params = log_system_params
        self._capture_terminal_logs = capture_terminal_logs

        self._run = None
        self._run_hash = run_hash

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        metric_items: Dict[str:Any] = {k: v for k, v in metrics.items()}

        if 'epoch' in metric_items:
            epoch: int = metric_items.pop('epoch')
        else:
            epoch = None

        for k, v in metric_items.items():
            name, context = self.parse_context(k)
            self.experiment.track(v, name=name, step=step, epoch=epoch, context=context)

    def parse_context(self, name):
        context = {}

        for ctx, mappings in self._context_prefixes.items():
            for category, prefix in mappings.items():
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    context[ctx] = category
                    break  # avoid prefix rename cascade

        for ctx, mappings in self._context_postfixes.items():
            for category, postfix in mappings.items():
                if name.endswith(postfix):
                    name = name[: -len(postfix)]
                    context[ctx] = category
                    break  # avoid postfix rename cascade

        return name, context

