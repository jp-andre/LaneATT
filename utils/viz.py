import argparse

import cv2
import torch
import random
import numpy as np
import torchvision.transforms as transforms

from lib.config import Config

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a dataset")
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--image", required=True)
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


def filter_good_lanes(lanes, tolerance=0.05):
    MAX_CANDIDATES = 20
    good_lanes = []
    tolerance = 0.05
    start_positions = np.array([-1])

    for lane in lanes[0][:MAX_CANDIDATES]:
        startx = lane.points[0,0]
        diffs = np.abs(start_positions - startx) > tolerance
        if np.all(diffs):
            print("Found new lane starting at", lane.points[0])
            good_lanes.append(lane)
            start_positions = np.append(start_positions, startx)

    return good_lanes


def main():
    args = parse_args()
    cfg = Config(args.cfg)

    print("Loading inference model: LaneATT")

    device = torch.device("cuda:0")
    model = cfg.get_model()

    model_path = "experiments/custom/models/model_0100.pt"
    train_state = torch.load(model_path)

    model.load_state_dict(train_state["model"])
    model = model.to(device)
    model.eval()
    print("Successfully loaded inference model")

    image = cv2.imread(args.image)
    img = cv2.resize(image, (640, 360), interpolation=cv2.INTER_LINEAR)

    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0)

    _, lanes = infer(model, tensor, device)
    good_lanes = filter_good_lanes(lanes)

    for lane in good_lanes:
        draw_lane(img, lane, 640, 360, color=(0, 255, 0))
    cv2_showimage(img)


if __name__ == "__main__":
    main()
