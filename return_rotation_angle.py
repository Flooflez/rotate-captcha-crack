import argparse

import torch
from PIL import Image

from rotate_captcha_crack.common import device
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha

if __name__ == "__main__":

    with torch.no_grad():
        model = RotNetR(train=False, cls_num=180)
        model_path = WhereIsMyModel(model).with_index(-1).model_dir / "best.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))) #EDITED
        model = model.to(device=device)
        model.eval()

        img = Image.open("datasets/maxmyta/test5.jpg") #CHANGE DIR TO CHECK IMAGE
        img_ts = process_captcha(img)
        img_ts = img_ts.to(device=device)

        predict = model.predict(img_ts)
        degree = predict * 360
        print(f"Predict degree: {degree:.4f}°")
