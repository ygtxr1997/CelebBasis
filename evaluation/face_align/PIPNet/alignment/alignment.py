import cv2
import numpy as np
from skimage import transform as trans


def get_center(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return np.array([centroid])


def extract_five_lmk(lmk):
    x = lmk[..., :2]
    left_eye = get_center(x[36:42])
    right_eye = get_center(x[42:48])
    nose = x[30:31]
    left_mouth = x[48:49]
    right_mouth = x[54:55]
    x = np.concatenate([left_eye, right_eye, nose, left_mouth, right_mouth], axis=0)
    return x


set1 = np.array(
    [
        [41.125, 50.75],
        [71.75, 49.4375],
        [49.875, 73.0625],
        [45.9375, 87.9375],
        [70.4375, 87.9375],
    ],
    dtype=np.float32,
)

arcface_src = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


ffhq = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

mtcnn = np.array(
    [
        [40.95041, 52.341854],
        [70.90203, 52.17619],
        [56.02142, 69.376114],
        [43.716904, 86.910675],
        [68.52042, 86.77348],
    ],
    dtype=np.float32,
)

arcface_src = np.expand_dims(arcface_src, axis=0)
set1 = np.expand_dims(set1, axis=0)
ffhq = np.expand_dims(ffhq, axis=0)
mtcnn = np.expand_dims(mtcnn, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode="set1"):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")
    if mode == "arcface":
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    elif mode == "set1":
        if image_size == 112:
            src = set1
        else:
            src = float(image_size) / 112 * set1
    elif mode == "ffhq":
        if image_size == 512:
            src = ffhq
        else:
            src = float(image_size) / 512 * ffhq
    elif mode == "mtcnn":
        if image_size == 112:
            src = mtcnn
        else:
            src = float(image_size) / 112 * mtcnn
    else:
        print("no mode like {}".format(mode))
        exit()
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def estimate_norm_any(lmk_from, lmk_to, image_size=112):
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk_from, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float("inf")
    src = lmk_to[np.newaxis, ...]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk_from, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode="arcface", borderValue=0.0):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=borderValue)
    return warped


def norm_crop_with_M(img, landmark, image_size=112, mode="arcface", borderValue=0.0):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=borderValue)
    return warped, M


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[: resized_im.shape[0], : resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    # translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2, output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data, M, (output_size, output_size), borderValue=0.0)
    return cropped, M


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    # print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        # print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


def paste_back(img, mat, ori_img):
    mat_rev = np.zeros([2, 3])
    div1 = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    mat_rev[0][0] = mat[1][1] / div1
    mat_rev[0][1] = -mat[0][1] / div1
    mat_rev[0][2] = -(mat[0][2] * mat[1][1] - mat[0][1] * mat[1][2]) / div1
    div2 = mat[0][1] * mat[1][0] - mat[0][0] * mat[1][1]
    mat_rev[1][0] = mat[1][0] / div2
    mat_rev[1][1] = -mat[0][0] / div2
    mat_rev[1][2] = -(mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2]) / div2

    img_shape = (ori_img.shape[1], ori_img.shape[0])

    img = cv2.warpAffine(img, mat_rev, img_shape)
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
    ori_img = img_mask * img + (1 - img_mask) * ori_img
    ori_img = ori_img.astype(np.uint8)
    return ori_img
