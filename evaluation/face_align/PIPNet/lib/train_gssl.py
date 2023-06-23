import cv2, os
import sys
sys.path.insert(0, '..')
import numpy as np
from PIL import Image
import logging
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from networks_gssl import *
import data_utils_gssl
from functions_gssl import * 

if not len(sys.argv) == 2:
    print('Format:')
    print('python lib/train_gssl.py config_file')
    exit(0)
experiment_name = sys.argv[1].split('/')[-1][:-3]
data_name = sys.argv[1].split('/')[-2]
config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

my_config = importlib.import_module(config_path, package='PIPNet')
Config = getattr(my_config, 'Config')
cfg = Config()
cfg.experiment_name = experiment_name
cfg.data_name = data_name

os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists(os.path.join('./snapshots', cfg.data_name)):
    os.mkdir(os.path.join('./snapshots', cfg.data_name))
save_dir = os.path.join('./snapshots', cfg.data_name, cfg.experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(os.path.join('./logs', cfg.data_name)):
    os.mkdir(os.path.join('./logs', cfg.data_name))
log_dir = os.path.join('./logs', cfg.data_name, cfg.experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)

print('###########################################')
print('experiment_name:', cfg.experiment_name)
print('data_name:', cfg.data_name)
print('det_head:', cfg.det_head)
print('net_stride:', cfg.net_stride)
print('batch_size:', cfg.batch_size)
print('init_lr:', cfg.init_lr)
print('num_epochs:', cfg.num_epochs)
print('decay_steps:', cfg.decay_steps)
print('input_size:', cfg.input_size)
print('backbone:', cfg.backbone)
print('pretrained:', cfg.pretrained)
print('criterion_cls:', cfg.criterion_cls)
print('criterion_reg:', cfg.criterion_reg)
print('cls_loss_weight:', cfg.cls_loss_weight)
print('reg_loss_weight:', cfg.reg_loss_weight)
print('num_lms:', cfg.num_lms)
print('save_interval:', cfg.save_interval)
print('num_nb:', cfg.num_nb)
print('use_gpu:', cfg.use_gpu)
print('gpu_id:', cfg.gpu_id)
print('curriculum:', cfg.curriculum)
print('###########################################')
logging.info('###########################################')
logging.info('experiment_name: {}'.format(cfg.experiment_name))
logging.info('data_name: {}'.format(cfg.data_name))
logging.info('det_head: {}'.format(cfg.det_head))
logging.info('net_stride: {}'.format(cfg.net_stride))
logging.info('batch_size: {}'.format(cfg.batch_size))
logging.info('init_lr: {}'.format(cfg.init_lr))
logging.info('num_epochs: {}'.format(cfg.num_epochs))
logging.info('decay_steps: {}'.format(cfg.decay_steps))
logging.info('input_size: {}'.format(cfg.input_size))
logging.info('backbone: {}'.format(cfg.backbone))
logging.info('pretrained: {}'.format(cfg.pretrained))
logging.info('criterion_cls: {}'.format(cfg.criterion_cls))
logging.info('criterion_reg: {}'.format(cfg.criterion_reg))
logging.info('cls_loss_weight: {}'.format(cfg.cls_loss_weight))
logging.info('reg_loss_weight: {}'.format(cfg.reg_loss_weight))
logging.info('num_lms: {}'.format(cfg.num_lms))
logging.info('save_interval: {}'.format(cfg.save_interval))
logging.info('num_nb: {}'.format(cfg.num_nb))
logging.info('use_gpu: {}'.format(cfg.use_gpu))
logging.info('gpu_id: {}'.format(cfg.gpu_id))
logging.info('###########################################')

if cfg.curriculum:
    # self-training with curriculum
    task_type_list = ['cls3', 'cls2', 'std', 'std', 'std']
else:
    # standard self-training 
    task_type_list = ['std']*3

meanface_indices, reverse_index1, reverse_index2, max_len = get_meanface(os.path.join('data', cfg.data_name, 'meanface.txt'), cfg.num_nb)

if cfg.det_head == 'pip':
    if cfg.backbone == 'resnet18':
        resnet18 = models.resnet18(pretrained=cfg.pretrained)
        net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
    else:
        print('No such backbone!')
        exit(0)
else:
    print('No such head:', cfg.det_head)
    exit(0)

if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

criterion_cls = None
if cfg.criterion_cls == 'l2':
    criterion_cls = nn.MSELoss(reduction='sum')
elif cfg.criterion_cls == 'l1':
    criterion_cls = nn.L1Loss()
else:
    print('No such cls criterion:', cfg.criterion_cls)

criterion_reg = None
if cfg.criterion_reg == 'l1':
    criterion_reg = nn.L1Loss(reduction='sum')
elif cfg.criterion_reg == 'l2':
    criterion_reg = nn.MSELoss()
else:
    print('No such reg criterion:', cfg.criterion_reg)

points_flip = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, 65, 64, 63, 62, 61, 68, 67, 66]
points_flip = (np.array(points_flip)-1).tolist()
assert len(points_flip) == 68

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)

labels = get_label(cfg.data_name, 'train_300W.txt', 'std')

train_data = data_utils_gssl.ImageFolder_pip(os.path.join('data', cfg.data_name, 'images_train'), 
                                        labels, cfg.input_size, cfg.num_lms, 
                                        cfg.net_stride, points_flip, meanface_indices,
                                        transforms.Compose([
                                        transforms.RandomGrayscale(0.2),
                                        transforms.ToTensor(),
                                        normalize]))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

train_model(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)

###############
# test
norm_indices = [36, 45]

preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
test_data_list = ['300W', 'COFW', 'WFLW']
for test_data in test_data_list:
    labels = get_label(cfg.data_name, 'test_'+test_data+'.txt')
    nmes = []
    norm = None
    for label in labels:
        image_name = label[0]
        lms_gt = label[1]
        image_path = os.path.join('data', cfg.data_name, 'images_test_'+test_data, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (cfg.input_size, cfg.input_size))
        inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
        inputs = preprocess(inputs).unsqueeze(0)
        inputs = inputs.to(device)
        lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
        # inter-ocular
        norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])
        #############################
        # merge neighbor predictions
        lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten().cpu().numpy()
        tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
        tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
        tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()
        #############################
        nme = compute_nme(lms_pred_merge, lms_gt, norm)
        nmes.append(nme)
    
    print('{} nme: {}'.format(test_data, np.mean(nmes)))
    logging.info('{} nme: {}'.format(test_data, np.mean(nmes)))

for ti, task_type in enumerate(task_type_list):
    print('###################################################')
    print('Iter:', ti, 'task_type:', task_type)
    ###############
    # estimate
    if cfg.data_name == 'data_300W_COFW_WFLW':
        est_data_list = ['COFW', 'WFLW']
    elif cfg.data_name == 'data_300W_CELEBA': 
        est_data_list = ['CELEBA']
    else:
        print('No such data!')
        exit(0)
    est_preds = []
    for est_data in est_data_list:
        labels = get_label(cfg.data_name, 'train_'+est_data+'.txt')
        for label in labels:
            image_name = label[0]
            #print(image_name)
            image_path = os.path.join('data', cfg.data_name, 'images_train', image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (cfg.input_size, cfg.input_size))
            inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
            #############################
            # merge neighbor predictions
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten().cpu().numpy()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()
            #############################
            est_preds.append([image_name, task_type, lms_pred_merge])
    
    ################
    # GSSL 
    if cfg.det_head == 'pip':
        if cfg.backbone == 'resnet18':
            resnet18 = models.resnet18(pretrained=cfg.pretrained)
            net = Pip_resnet18(resnet18, cfg.num_nb, num_lms=cfg.num_lms, input_size=cfg.input_size, net_stride=cfg.net_stride)
        else:
            print('No such backbone!')
            exit(0)
    else:
        print('No such head:', cfg.det_head)
        exit(0)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)
    labels = get_label(cfg.data_name, 'train_300W.txt', 'std')
    labels += est_preds
    
    train_data = data_utils_gssl.ImageFolder_pip(os.path.join('data', cfg.data_name, 'images_train'), 
                                            labels, cfg.input_size, cfg.num_lms, 
                                            cfg.net_stride, points_flip, meanface_indices,
                                            transforms.Compose([
                                            transforms.RandomGrayscale(0.2),
                                            transforms.ToTensor(),
                                            normalize]))
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    
    train_model(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight, cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)
    
    ###############
    # test
    preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])
    test_data_list = ['300W', 'COFW', 'WFLW']
    for test_data in test_data_list:
        labels = get_label(cfg.data_name, 'test_'+test_data+'.txt')
        nmes = []
        norm = None
        for label in labels:
            image_name = label[0]
            lms_gt = label[1]
            image_path = os.path.join('data', cfg.data_name, 'images_test_'+test_data, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (cfg.input_size, cfg.input_size))
            inputs = Image.fromarray(image[:,:,::-1].astype('uint8'), 'RGB')
            inputs = preprocess(inputs).unsqueeze(0)
            inputs = inputs.to(device)
            lms_pred_x, lms_pred_y, lms_pred_nb_x, lms_pred_nb_y, outputs_cls, max_cls = forward_pip(net, inputs, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb)
            # inter-ocular
            norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])
            #############################
            # merge neighbor predictions
            lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten().cpu().numpy()
            tmp_nb_x = lms_pred_nb_x[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_nb_y = lms_pred_nb_y[reverse_index1, reverse_index2].view(cfg.num_lms, max_len)
            tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1,1)
            tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1,1)
            lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten().cpu().numpy()
            #############################
            nme = compute_nme(lms_pred_merge, lms_gt, norm)
            nmes.append(nme)
        
        print('{} nme: {}'.format(test_data, np.mean(nmes)))
        logging.info('{} nme: {}'.format(test_data, np.mean(nmes)))


