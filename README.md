# For quick testing

To run inference on a single image:
```
PYTHONPATH=. python utils/viz.py --image datasets/tusimple-test/clips/0531/1492628912583733966/10.jpg
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
