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
# os.system("curl -o LaneATT-main.zip https://github.com/jp-andre/LaneATT/archive/refs/heads/main.zip")
os.system("unzip -qo LaneATT-main.zip && mv LaneATT-main/* .")
os.system("pip install -r requirements.txt")
sys.path.append(".")
importlib.invalidate_caches()

# 3. Load LaneATT module and model
from lib.viz import load_model, img2tensor, filter_good_lanes, draw_lane, infer
model, device = load_model(pt_filename)

global shared_objects
shared_objects = {}

print(f"Initialization of LaneATT completed!")
