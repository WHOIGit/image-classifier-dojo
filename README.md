# image-classifier-dojo
Great for training computer vision classifier models!

## installation

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


