import os
import sys
import importlib
from model_helpers.log import log_info

# 1. Download pretrained weights
pt_filename = "laneatt-weights.pt"
cachefile = f"~/.cache/{pt_filename}"
url = "https://collimator-devops-resources.s3.us-west-2.amazonaws.com/ml-demos/LaneATT/model_0100.pt"

if not os.path.exists(cachefile):
    print("Downloading pretrained weights...")
    os.system(f"curl -o {cachefile} {url}")

os.system(f"ln -sf {cachefile} {pt_filename}")

# 2. Prepare code for LaneATT
os.system(f"unzip LaneATT.zip")
os.system(f"pip install -r LaneATT.zip")
sys.path.append(".")
importlib.invalidate_caches()

# 3. Load LaneATT module and model
from lib.viz import load_model, img2tensor, filter_good_lanes, draw_lane, infer
model, device = load_model(pt_filename)

global shared_objects
shared_objects = {}

log_info(f"Initialization of LaneATT completed!")
