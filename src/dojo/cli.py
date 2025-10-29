from dojo.schemas import *

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliSubCommand

class Trainables(BaseModel):
    MULTICLASS: CliSubCommand[TrainingRunConfig] = Field(
        ..., description="Train a Multiclass Image Classifier"
    )
    #multioutput: CliSubCommand[str] = Field(description='NOT IMPLEMENTED')
    #selfsupervised: CliSubCommand[str] = Field(description='NOT IMPLEMENTED')
    #homo_ensemble: CliSubCommand[str] = Field(description='NOT IMPLEMENTED')

class Settings(BaseSettings, cli_parse_args=True, case_sensitive=True, cli_prog_name='image_classifier_dojo'):
    TRAIN: CliSubCommand[Trainables] = Field(..., description="Perform Training")
    #utils: CliSubCommand[str] = Field(..., description="NOT IMPLEMENTED. Utility Commands")

def main():
#     import sys
#     sys.argv = '''src/dojo/cli.py
# TRAIN
# MULTICLASS
# --logger.repo=.aim
# --logger.context_postfixes={"averaging":{"macro":"_macro","weighted":"_weighted","micro":"_micro","None":"_perclass"}}
# --dataset_config.dataset.classlist=datasets/miniset_labels.list
# --dataset_config.dataset.trainlist=datasets/miniset_training.list
# --dataset_config.dataset.vallist=datasets/miniset_validation.list
# --dataset_config.training_transforms.flip=xy
# --model.backbone.model_name=efficientnet_b0
# --model.backbone.pretrained_weights=DEFAULT
# --model.head.head_type=multiclass
# --runtime.fast_dev_run=true'''.split('\n')
#    """python src/dojo/cli.py TRAIN MULTICLASS --logger '{"repo": ".aim", "experiment": null, "run": null, "note": null, "artifacts_location": null, "context_prefixes": null, "context_postfixes": {"averaging": {"macro": "_macro", "weighted": "_weighted", "micro": "_micro", "None": "_perclass"}}}' --dataset_config '{"dataset": {"classlist": "datasets/miniset_labels.list", "trainlist": "datasets/miniset_training.list", "vallist": "datasets/miniset_validation.list", "testlist": null}, "img_norm": null, "training_transforms": {"flip": "xy"}}' --model '{"backbone": {"model_name": "efficientnet_b0", "pretrained_weights": "DEFAULT"}, "head": {"head_type": "multiclass", "name": "head", "num_classes": null}, "weights": null}' --training '{"epochs": {"max_epochs": 100, "min_epochs": null, "patience": null}, "batch_size": 256, "model_optims": {"loss_config": {"loss_function": "CrossEntropyLoss", "reduction": "sum", "label_smoothing": 0.0, "weight": null}, "optimizer_config": {"optimizer": "Adam", "lr": 0.001, "amsgrad": false}, "freeze": null}, "precision": "16-mixed", "swa": null}' --runtime '{"experiment": null, "run_name": "loose-tortoise", "checkpoint_callback_config": {"dirpath": "./experiments/loose-tortoise", "filename": "loss-{val_loss:3.3f}_ep-{epoch:03.0f}", "monitor": "val_loss", "save_last": "link", "save_top_k": 3, "auto_insert_metric_name": false}, "seed": 600340444, "num_workers": 4, "autobatch": null, "fast_dev_run": true, "onnx_callback_configs": []}'"""
    args = Settings()
    if args.TRAIN:
        if args.TRAIN.MULTICLASS:
            from dojo.multiclass.train import main as train_multiclass
            train_multiclass(args.TRAIN.MULTICLASS)
        # elif args.TRAIN.MULTIOUTPUT:
        #     train_multioutput(args.TRAIN.MULTIOUTPUT)
        # elif args.TRAIN.SSL:
        #     train_ssl(args.TRAIN.SSL)
        # elif args.TRAIN.HOMO_ENSEMBLE:
        #     train_homoensemble(args.TRAIN.HOMO_ENSEMBLE)
        else:
            raise NotImplementedError
    # elif args.UTIL: # this stuff is actual in src/tools
    #     ... # TODO get listfiles from directory
    #     ... # TODO calculate img_norm, produce json file for config
    #     ... # TODO onnx utils
    else:
        raise NotImplementedError

if __name__ == '__main__':
    main()