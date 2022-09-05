import os
import sys
import importlib

# 1. Download pretrained weights
pt_filename = "model_0100.pt"
homedir = os.environ["HOME"]
cachefile = f"{homedir}/.cache/{pt_filename}"
url = "https://collimator-devops-resources.s3.us-west-2.amazonaws.com/ml-demos/LaneATT/model_0100.pt"

if not os.path.exists(cachefile):
    print("Downloading pretrained weights...")
    os.system(f"curl -o {cachefile} {url}")

os.system(f"ln -sf {cachefile} {pt_filename}")

# 2. Prepare code for LaneATT
if not os.path.exists("LaneATT-main.zip"):
    os.system("curl -Lo LaneATT-main.zip https://github.com/jp-andre/LaneATT/archive/refs/heads/main.zip")
os.system("unzip -qo LaneATT-main.zip && mv LaneATT-main/* .")
os.system("pip install -r requirements.txt")
os.system("cd lib/nms && pip install .")
sys.path.append(".")
importlib.invalidate_caches()

# 3. Load LaneATT module and model
from utils.viz import load_model
model, device = load_model(pt_filename)

global shared_objects
shared_objects = {
    "model": model
}

print(f"Initialization of LaneATT completed!")
