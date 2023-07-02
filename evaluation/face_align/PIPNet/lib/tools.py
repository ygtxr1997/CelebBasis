import cv2
import sys


from math import floor
from evaluation.face_align.PIPNet.FaceBoxesV2.faceboxes_detector import *

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models

from evaluation.face_align.PIPNet.lib.networks import *
from evaluation.face_align.PIPNet.lib.functions import *
from evaluation.face_align.PIPNet.reverse_index import ri1, ri2


class Config:
    def __init__(self):
        self.det_head = "pip"
        self.net_stride = 32
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [30, 50]
        self.input_size = 256
        self.backbone = "resnet101"
        self.pretrained = True
        self.criterion_cls = "l2"
        self.criterion_reg = "l1"
        self.cls_loss_weight = 10
        self.reg_loss_weight = 1
        self.num_lms = 98
        self.save_interval = self.num_epochs
        self.num_nb = 10
        self.use_gpu = True
        self.gpu_id = 3


def get_lmk_model():

    cfg = Config()

    resnet101 = models.resnet101(pretrained=cfg.pretrained)
    net = Pip_resnet101(
        resnet101,
        cfg.num_nb,
        num_lms=cfg.num_lms,
        input_size=cfg.input_size,
        net_stride=cfg.net_stride,
    )

    if cfg.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    net = net.to(device)

    weight_file = 'evaluation/face_align/PIPNet/weights/epoch59.pth'
    state_dict = torch.load(weight_file, map_location=device)
    net.load_state_dict(state_dict)

    detector = FaceBoxesDetector(
        "FaceBoxes",
        "evaluation/face_align/PIPNet/weights/FaceBoxesV2.pth",
        use_gpu=True,
        device="cuda:0",
    )
    return net, detector


def demo_image(
    image_file,
    net,
    detector,
    input_size=256,
    net_stride=32,
    num_nb=10,
    use_gpu=True,
    device="cuda:0",
):

    my_thresh = 0.6
    det_box_scale = 1.2
    net.eval()
    preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    reverse_index1, reverse_index2, max_len = ri1, ri2, 17
    # image = cv2.imread(image_file)
    image = image_file
    image_height, image_width, _ = image.shape
    detections, _ = detector.detect(image, my_thresh, 1)
    lmks = []
    for i in range(len(detections)):
        det_xmin = detections[i][2]
        det_ymin = detections[i][3]
        det_width = detections[i][4]
        det_height = detections[i][5]
        det_xmax = det_xmin + det_width - 1
        det_ymax = det_ymin + det_height - 1

        det_xmin -= int(det_width * (det_box_scale - 1) / 2)
        # remove a part of top area for alignment, see paper for details
        det_ymin += int(det_height * (det_box_scale - 1) / 2)
        det_xmax += int(det_width * (det_box_scale - 1) / 2)
        det_ymax += int(det_height * (det_box_scale - 1) / 2)
        det_xmin = max(det_xmin, 0)
        det_ymin = max(det_ymin, 0)
        det_xmax = min(det_xmax, image_width - 1)
        det_ymax = min(det_ymax, image_height - 1)
        det_width = det_xmax - det_xmin + 1
        det_height = det_ymax - det_ymin + 1

        # cv2.rectangle(image, (det_xmin, det_ymin), (det_xmax, det_ymax), (0, 0, 255), 2)

        det_crop = image[det_ymin:det_ymax, det_xmin:det_xmax, :]
        det_crop = cv2.resize(det_crop, (input_size, input_size))
        inputs = Image.fromarray(det_crop[:, :, ::-1].astype("uint8"), "RGB")
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)
        (
            lms_pred_x,
            lms_pred_y,
            lms_pred_nb_x,
            lms_pred_nb_y,
            outputs_cls,
            max_cls,
        ) = forward_pip(net, inputs, preprocess, input_size, net_stride, num_nb)
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(98, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(98, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        lms_pred = lms_pred.cpu().numpy()
        lms_pred_merge = lms_pred_merge.cpu().numpy()
        lmk_ = []
        for i in range(98):
            x_pred = lms_pred_merge[i * 2] * det_width
            y_pred = lms_pred_merge[i * 2 + 1] * det_height

            # cv2.circle(
            #     image,
            #     (int(x_pred) + det_xmin, int(y_pred) + det_ymin),
            #     1,
            #     (0, 0, 255),
            #     1,
            # )

            lmk_.append([int(x_pred) + det_xmin, int(y_pred) + det_ymin])
        lmks.append(np.array(lmk_))

    # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("./1_out.jpg", image_bgr)

    return lmks


if __name__ == "__main__":
    net, detector = get_lmk_model()
    demo_image(
        "/apdcephfs/private_ahbanliang/codes/Real-ESRGAN-master/tmp_frames/yanikefu/frame00000046.png",
        net,
        detector,
    )
