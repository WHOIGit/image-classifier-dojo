# Use a lightweight Python image
FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

## Include demo dataset
#COPY datasets/miniset/ ./datasets/miniset/
#COPY datasets/miniset_*.list ./datasets/

# Install project dependencies
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install .[train]

# Set the default command to run your training script
ENTRYPOINT ["python", "src/dojo/cli.py"]
