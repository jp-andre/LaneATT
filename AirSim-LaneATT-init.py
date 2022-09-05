import os
import sys
import importlib

print(f"Prepraring environment for LaneATT")

# 1. Download pretrained weights
pt_filename = "model_0100.pt"
homedir = os.environ["HOME"]
cachefile = f"{homedir}/.cache/{pt_filename}"
url = "https://collimator-devops-resources.s3.us-west-2.amazonaws.com/ml-demos/LaneATT/model_0100.pt"

if not os.path.exists(cachefile):
    print("Downloading pretrained weights...")
    os.system(f"curl -qo {cachefile} {url}")

os.system(f"ln -sf {cachefile} {pt_filename}")

# 2. Prepare code for LaneATT
if not os.path.exists("LaneATT-main.zip"):
    os.system("curl -Lo LaneATT-main.zip https://github.com/jp-andre/LaneATT/archive/refs/heads/main.zip")
os.system("unzip -qo LaneATT-main.zip && mv -f LaneATT-main/* .")
# os.system("pip install -r LaneATT-main/requirements.txt")
# os.system("cd LaneATT-main/lib/nms && pip install .")
sys.path.append("LaneATT-main")

# 3. Load LaneATT module and model
# from utils.viz import load_model
# model, device = load_model(pt_filename)

# 4. Prepare AirSim
# os.system('pip install msgpack-rpc-python')
# os.system('pip install airsim')
# os.system('pip install pillow')

# All good!

importlib.invalidate_caches()
print(f"Initialization of LaneATT completed!")

global shared_objects
shared_objects = {}
