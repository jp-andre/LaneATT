import argparse

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import similaritymeasures

from lib.models.laneatt import LaneATT


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a dataset")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="model_0100.pt", required=False)
    args = parser.parse_args()

    return args


def infer(model, tensor, device):
    with torch.no_grad():
        tensor = tensor.to(device)
        infered = model(tensor)
        outputs = model.decode(infered)
        # print("outputs", outputs)

        as_lanes = model.decode(infered, as_lanes=True)
        # print("as lanes", as_lanes)

        return outputs, as_lanes


def cv2_showimage(img):
    cv2.imshow("img", img)

    while cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(delay=200)
        if (keyCode & 0xFF) == ord("q"):
            break

    cv2.destroyAllWindows()


def draw_lane(img, lane, w, h, color=(0, 255, 0)):
    points = lane.points
    for pt1, pt2 in zip(points[:-1], points[1:]):

        x1 = int(pt1[0] * w)
        x2 = int(pt2[0] * w)
        y1 = int(pt1[1] * h)
        y2 = int(pt2[1] * h)

        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=2)


def filter_good_lanes(outputs, lanes, tolerance=0.05):
    MAX_CANDIDATES = 20
    good_lanes = []
    tolerance = 0.1

    # Look for lanes that are distinct enough
    # in the first few candidates (the highest probability of being lanes)
    for k in range(MAX_CANDIDATES):
        lane = lanes[0][k]

        skip = False
        for cmp in good_lanes:
            p1 = cmp.points[:20]
            p2 = lane.points[:20]
            dist = similaritymeasures.area_between_two_curves(p1, p2)
            if dist < tolerance*tolerance:
                skip = True
                break

        if not skip:
            print("Found new lane starting at", lane.points[0])
            good_lanes.append(lane)

    return good_lanes


def load_pretrained_weights(weights_filename):
    if not os.path.exists(weights_filename):
        print("Pretrained weights not found, downloading...")
        url = f"https://collimator-devops-resources.s3.us-west-2.amazonaws.com/ml-demos/LaneATT/{weights_filename}"
        os.system(f"wget {url}")

    train_state = torch.load(weights_filename)
    return train_state['model']

def load_model(weights_filename = "model_0100.pt"):
    print("Loading inference model: LaneATT (might require CUDA)")

    device = torch.device("cuda:0")

    params = {
        "backbone": "resnet34",
        "S": 72,
        "topk_anchors": 1000,
        "anchors_freq_path": "data/tusimple_anchors_freq.pt",
        "img_h": 360,
        "img_w": 640,
    }

    model = LaneATT(**params)

    weights = load_pretrained_weights(weights_filename)
    model.load_state_dict(weights)
    model = model.to(device)
    model.eval()
    print("Successfully loaded inference model")

    return model, device


def img2tensor(image):
    img = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)
    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)
    return tensor


def main():
    args = parse_args()

    model, device = load_model(args.weights)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = img2tensor(image)

    outputs, lanes = infer(model, tensor, device)
    good_lanes = filter_good_lanes(outputs, lanes)

    for lane in good_lanes:
        draw_lane(image, lane, 640, 360, color=(0, 255, 0))
    cv2_showimage(image)


if __name__ == "__main__":
    main()
