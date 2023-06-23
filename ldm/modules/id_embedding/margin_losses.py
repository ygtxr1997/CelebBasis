import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import Parameter

import numpy as np

__all__ = ['Softmax', 'AMCosFace', 'AMArcFace', ]


MIN_NUM_PATCHES = 16


""" All losses can run in 'torch.distributed.DistributedDataParallel'.
"""

class Softmax(nn.Module):
    r"""Implementation of Softmax (normal classification head):
        Args:
            in_features: dimension (d_in) of input feature (B, d_in)
            out_features: dimension (d_out) of output feature (B, d_out)
            device_id: the ID of GPU where the model will be trained by data parallel (or DP). (not used)
                        if device_id=None, it will be trained on model parallel (or DDP). (recommend!)
        """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device_id,
                 ):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, embedding, label):
        """
        :param embedding: learned face representation
        :param label:
            - label >= 0: ground truth identity
            - label = -1: invalid identity for this GPU (refer to 'PartialFC')
            + Example: label = torch.tensor([-1, 4, -1, 5, 3, -1])
        :return:
        """
        if self.device_id is None:
            """ Regular linear layer.
            """
            out = F.linear(embedding, self.weight, self.bias)
        else:
            raise ValueError('DataParallel is not implemented yet.')
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out


""" Not Used """
class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        print('ArcFace, s=%.1f, m=%.2f' % (s, m))

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        else:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


""" Not Used """
class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.4):
        super(CosFace, self).__init__()
        print('CosFace, s=%.1f, m=%.2f' % (s, m))
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).cuda()
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
            one_hot.scatter_(1, label.cuda(self.device_id[0]).view(-1, 1).long(), 1)
        else:
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


class AMCosFace(nn.Module):
    r"""Implementation of Adaptive Margin CosFace:
    cos(theta)-m+k(theta-a)
    When k is 0, AMCosFace degenerates into CosFace.
    Args:
        in_features: dimension (d_in) of input feature (B, d_in)
        out_features: dimension (d_out) of output feature (B, d_out)
        device_id: the ID of GPU where the model will be trained by data parallel (or DP). (not used)
                    if device_id=None, it will be trained on model parallel (or DDP). (recommend!)
        s: norm of input feature
        m: margin
        a: AM Loss
        k: AM Loss
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device_id,
                 s: float = 64.0,
                 m: float = 0.4,
                 a: float = 1.2,
                 k: float = 0.1,
                 ):
        super(AMCosFace, self).__init__()
        print('AMCosFace, s=%.1f, m=%.2f, a=%.2f, k=%.2f' % (s, m, a, k))
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        self.a = a
        self.k = k

        """ Weight Matrix W (d_out, d_in) """
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        """
        :param embedding: learned face representation
        :param label:
            - label >= 0: ground truth identity
            - label = -1: invalid identity for this GPU (refer to 'PartialFC')
            + Example: label = torch.tensor([-1, 4, -1, 5, 3, -1])
        :return:
        """
        if self.device_id is None:
            """ - embedding: shape is (B, d_in)
                - weight: shape is (d_out, d_in)
                - cosine: shape is (B, d_out)
                + F.normalize is very important here.
            """
            cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))  # y = xA^T + b
        else:
            raise ValueError('DataParallel is not implemented yet.')
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x),
                                                     F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)

        """ - index: the index of valid identity in label, shape is (d_valid, )
            + torch.where() returns a tuple indicating the index of each dimension
            + Example: index = torch.tensor([1, 3, 4])
        """
        index = torch.where(label != -1)[0]

        """ - m_hot: one-hot tensor of margin m_2, shape is (d_valid, d_out)
            + torch.tensor.scatter_(dim, index, source) is usually used to generate ont-hot tensor
            + Example: label = torch.tensor([-1, 4, -1, 5, 3, -1])
                       index = torch.tensor([1, 3, 4])  # d_valid = index.shape[0] = 3
                       m_hot = torch.tensor([[0, 0, 0, 0, m, 0],
                                             [0, 0, 0, 0, 0, m],
                                             [0, 0, 0, m, 0, 0],
                                            ])
        """
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)

        """ logit(theta) = cos(theta) - m_2 + k * (theta - a)
            - theta = cosine.acos_()
            + Example: m_hot = torch.tensor([[0, 0, 0, 0, m-k(theta[0,4]-a), 0],
                                             [0, 0, 0, 0, 0, m-k(theta[1,5]-a)],
                                             [0, 0, 0, m-k(theta[2,3]-a), 0, 0],
                                            ])
        """
        a = self.a
        k = self.k
        m_hot[range(0, index.size()[0]), label[index]] -= k * (cosine[index, label[index]].acos_() - a)
        cosine[index] -= m_hot

        """ Because we have used F.normalize, we should rescale the logit term by s.
        """
        output = cosine * self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) \
               + ', a = ' + str(self.a) \
               + ', k = ' + str(self.k) \
               + ')'


class AMArcFace(nn.Module):
    r"""Implementation of Adaptive Margin ArcFace:
    cos(theta+m-k(theta-a))
    When k is 0, AMArcFace degenerates into ArcFace.
    Args:
        in_features: dimension (d_in) of input feature (B, d_in)
        out_features: dimension (d_out) of output feature (B, d_out)
        device_id: the ID of GPU where the model will be trained by data parallel (or DP). (not used)
                    if device_id=None, it will be trained on model parallel (or DDP). (recommend!)
        s: norm of input feature
        m: margin
        a: AM Loss
        k: AM Loss
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 device_id,
                 s: float = 64.0,
                 m: float = 0.5,
                 a: float = 1.2,
                 k: float = 0.1,
                 ):
        super(AMArcFace, self).__init__()
        print('AMArcFace, s=%.1f, m=%.2f, a=%.2f, k=%.2f' % (s, m, a, k))
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m
        self.a = a
        self.k = k

        """ Weight Matrix W (d_out, d_in) """
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        """
        :param embedding: learned face representation
        :param label:
            - label >= 0: ground truth identity
            - label = -1: invalid identity for this GPU (refer to 'PartialFC')
            + Example: label = torch.tensor([-1, 4, -1, 5, 3, -1])
        :return:
        """
        if self.device_id is None:
            """ - embedding: shape is (B, d_in)
                - weight: shape is (d_out, d_in)
                - cosine: shape is (B, d_out)
                + F.normalize is very important here.
            """
            cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))  # y = xA^T + b
        else:
            raise ValueError('DataParallel is not implemented yet.')
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x),
                                                     F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)

        """ - index: the index of valid identity in label, shape is (d_valid, )
            + torch.where() returns a tuple indicating the index of each dimension
            + Example: index = torch.tensor([1, 3, 4])
        """
        index = torch.where(label != -1)[0]

        """ - m_hot: one-hot tensor of margin m_2, shape is (d_valid, d_out)
            + torch.tensor.scatter_(dim, index, source) is usually used to generate ont-hot tensor
            + Example: label = torch.tensor([-1, 4, -1, 5, 3, -1])
                       index = torch.tensor([1, 3, 4])  # d_valid = index.shape[0] = 3
                       m_hot = torch.tensor([[0, 0, 0, 0, m, 0],
                                             [0, 0, 0, 0, 0, m],
                                             [0, 0, 0, m, 0, 0],
                                            ])
        """
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)

        """ logit(theta) = cos(theta) - m_2 + k * (theta - a)
            - theta = cosine.acos_()
            + Example: m_hot = torch.tensor([[0, 0, 0, 0, m-k(theta[0,4]-a), 0],
                                             [0, 0, 0, 0, 0, m-k(theta[1,5]-a)],
                                             [0, 0, 0, m-k(theta[2,3]-a), 0, 0],
                                            ])
        """
        a = self.a
        k = self.k
        m_hot[range(0, index.size()[0]), label[index]] -= k * (cosine[index, label[index]].acos_() - a)

        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) \
               + ', a = ' + str(self.a) \
               + ', k = ' + str(self.k) \
               + ')'


if __name__ == '__main__':
    cosine = torch.randn(6, 8) / 100
    cosine[0][2] = 0.3
    cosine[1][4] = 0.4
    cosine[2][6] = 0.5
    cosine[3][5] = 0.6
    cosine[4][3] = 0.7
    cosine[5][0] = 0.8
    label = torch.tensor([-1, 4, -1, 5, 3, -1])

    # layer = AMCosFace(in_features=8,
    #                   out_features=8,
    #                   device_id=None,
    #                   m=0.35, s=1.0,
    #                   a=1.2, k=0.1)

    # layer = Softmax(in_features=8,
    #                 out_features=8,
    #                 device_id=None)

    layer = AMArcFace(in_features=8,
                      out_features=8,
                      device_id=None,
                      m=0.5, s=1.0,
                      a=1.2, k=0.1)

    logit = layer(cosine, label)
    logit = F.softmax(logit, dim=-1)

    from utils.vis_tensor import plot_tensor
    plot_tensor((cosine, logit),
                ('embedding', 'logit'),
                'AMArc.jpg')