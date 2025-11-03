# image-classifier-dojo
Great for training computer vision classifier models!

## Installation

### Development
```bash
git clone git@github.com:WHOIGit/image-classifier-dojo.git
cd image-classifier-dojo
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # or whatever cuda version you need
pip install -e .[train,ssl,homo]
```

### Regular installation
```bash
pip install 'image_classifier_dojo[train] @ git+https://github.com/WHOIGit/image_classifier_dojo.git@<tag>'
```

### Just schemas
```bash
pip install 'image_classifier_dojo @ git+https://github.com/WHOIGit/image_classifier_dojo.git@<tag>'
```

## Docker Commands
```bash
REPO=harbor-registry.whoi.edu/amplify
IMAGE=image_classifier_dojo
TAG=v0.M.n
docker build . -t $REPO/$IMAGE:latest -t $REPO/$IMAGE:$TAG
docker push $REPO/$IMAGE --all-tags
```

### Usage
```bash
docker run -it --rm --gpus all --shm-size 8G $REPO/$IMAGE:$TAG TRAIN MULTICLASS --help
# -->
docker run -it --rm --gpus all --shm-size 8G $REPO/$IMAGE:$TAG TRAIN MULTICLASS --logger '{...}' --dataset_config '{...}' --model '{...}' --training '{...}' --runtime '{...}'
```

## Usage
```bash
python -m image_classifier_dojo TRAIN MULTICLASS --help
# or
python -m dojo TRAIN MULTICLASS --help
```
For schemas:
```python
from image_classifier_dojo.schemas import TrainingRunConfig

```

