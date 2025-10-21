# image-classifier-dojo
Great for training computer vision classifier models!

# installation

```bash
git clone git@github.com:WHOIGit/image-classifier-dojo.git
cd image-classifier-dojo
python3 -m venv venv
source /venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126  # or whatever cuda version you need
pip install -e packages/schemas
pip install -e packages/image-classifier-dojo[all]
```

