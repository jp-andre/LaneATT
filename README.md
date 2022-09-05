# For quick testing

## Warning: **CUDA is required, macOS is not supported!**

To run inference on a single image:
```
PYTHONPATH=. python utils/viz.py --image image.jpg
```

# Installation

```
conda create -n laneatt python=3.8 -y
conda activate laneatt
conda install pytorch==1.6 torchvision -c pytorch
pip install -r requirements.txt
cd lib/nms; python setup.py install; cd -
```

# Original code

Go to the fork upstream at https://github.com/lucastabelini/LaneATT.
Great resources there.
