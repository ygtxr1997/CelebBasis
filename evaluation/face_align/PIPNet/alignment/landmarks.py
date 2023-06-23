import numpy as np
import cv2
from PIL import Image
# from TDDFA_V2.FaceBoxes import FaceBoxes
# from TDDFA_V2.TDDFA import TDDFA


def get_5_from_98(lmk):
    lefteye = (lmk[60] + lmk[64] + lmk[96]) / 3  # lmk[96]
    righteye = (lmk[68] + lmk[72] + lmk[97]) / 3  # lmk[97]
    nose = lmk[54]
    leftmouth = lmk[76]
    rightmouth = lmk[82]
    return np.array([lefteye, righteye, nose, leftmouth, rightmouth])


def get_center(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return np.array([centroid])


def get_lmk(img, tddfa, face_boxes):
    # 仅接受一个人的图像
    boxes = face_boxes(img)
    n = len(boxes)
    if n < 1:
        return None
    param_lst, roi_box_lst = tddfa(img, boxes)
    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
    x = ver_lst[0].transpose(1, 0)[..., :2]
    left_eye = get_center(x[36:42])
    right_eye = get_center(x[42:48])
    nose = x[30:31]
    left_mouth = x[48:49]
    right_mouth = x[54:55]
    x = np.concatenate([left_eye, right_eye, nose, left_mouth, right_mouth], axis=0)
    return x


def save(x, trick=None, use_post=False):
    """ Paste img to ori_img """
    img, mat, ori_img, save_path, img_mask = x
    if mat is None:
        print('[Warning] mat is None.')
        ori_img = ori_img.astype(np.uint8)
        Image.fromarray(ori_img).save(save_path)
        return

    mat_rev = np.zeros([2, 3])
    div1 = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    mat_rev[0][0] = mat[1][1] / div1
    mat_rev[0][1] = -mat[0][1] / div1
    mat_rev[0][2] = -(mat[0][2] * mat[1][1] - mat[0][1] * mat[1][2]) / div1
    div2 = mat[0][1] * mat[1][0] - mat[0][0] * mat[1][1]
    mat_rev[1][0] = mat[1][0] / div2
    mat_rev[1][1] = -mat[0][0] / div2
    mat_rev[1][2] = -(mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / div2

    img_shape = (ori_img.shape[1], ori_img.shape[0])  # (h,w)

    img = cv2.warpAffine(img, mat_rev, img_shape)

    if img_mask is None:
        ''' hanbang version of paste masks '''
        img_white = np.full((256, 256), 255, dtype=float)
        img_white = cv2.warpAffine(img_white, mat_rev, img_shape)
        img_white[img_white > 20] = 255
        img_mask = img_white

        kernel = np.ones((40, 40), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=2)

        kernel_size = (20, 20)
        blur_size = tuple(2 * j + 1 for j in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        img_mask /= 255
        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    else:
        ''' yuange version of paste masks '''
        img_mask = cv2.warpAffine(img_mask, mat_rev, img_shape)
        img_mask = np.expand_dims(img_mask, axis=-1)

    ori_img = img_mask * img + (1 - img_mask) * ori_img
    ori_img = ori_img.astype(np.uint8)

    if trick is not None:
        ori_img = trick.gpen(ori_img, use_post)

    Image.fromarray(ori_img).save(save_path)

    # img_mask = np.array((img_mask * 255), dtype=np.uint8).squeeze()
    # Image.fromarray(img_mask).save('img_mask.jpg')
