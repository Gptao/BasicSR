# coding=utf-8
# import importlib
# import torch
# from collections import OrderedDict
# from copy import deepcopy
# from os import path as osp
# from tqdm import tqdm
#
# from basicsr.models.archs import define_network
# from basicsr.models.base_model import BaseModel
# from basicsr.utils import get_root_logger, imwrite, tensor2img
#
# loss_module = importlib.import_module('basicsr.models.losses')
# metric_module = importlib.import_module('basicsr.metrics')
#
#
# class SRModel(BaseModel):
#     """Base SR model for single image super-resolution."""
#
#     def __init__(self, opt):
#         super(SRModel, self).__init__(opt)
#
#         # define network
#         self.net_g = define_network(deepcopy(opt['network_g']))
#         self.net_g = self.model_to_device(self.net_g)
#         self.print_network(self.net_g)
#
#         # load pretrained models
#         load_path = self.opt['path'].get('pretrain_network_g', None)
#         if load_path is not None:
#             self.load_network(self.net_g, load_path,
#                               self.opt['path'].get('strict_load_g', True))
#
#         if self.is_train:
#             self.init_training_settings()
#
#     def init_training_settings(self):
#         self.net_g.train()
#         train_opt = self.opt['train']
#
#         # define losses
#         if train_opt.get('pixel_opt'):
#             pixel_type = train_opt['pixel_opt'].pop('type')
#             cri_pix_cls = getattr(loss_module, pixel_type)
#             self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
#                 self.device)
#         else:
#             self.cri_pix = None
#
#         if train_opt.get('perceptual_opt'):
#             percep_type = train_opt['perceptual_opt'].pop('type')
#             cri_perceptual_cls = getattr(loss_module, percep_type)
#             self.cri_perceptual = cri_perceptual_cls(
#                 **train_opt['perceptual_opt']).to(self.device)
#         else:
#             self.cri_perceptual = None
#
#         if self.cri_pix is None and self.cri_perceptual is None:
#             raise ValueError('Both pixel and perceptual losses are None.')
#
#         # set up optimizers and schedulers
#         self.setup_optimizers()
#         self.setup_schedulers()
#
#     def setup_optimizers(self):
#         train_opt = self.opt['train']
#         optim_params = []
#         for k, v in self.net_g.named_parameters():
#             if v.requires_grad:
#                 optim_params.append(v)
#             else:
#                 logger = get_root_logger()
#                 logger.warning(f'Params {k} will not be optimized.')
#
#         optim_type = train_opt['optim_g'].pop('type')
#         if optim_type == 'Adam':
#             self.optimizer_g = torch.optim.Adam(optim_params,
#                                                 **train_opt['optim_g'])
#         else:
#             raise NotImplementedError(
#                 f'optimizer {optim_type} is not supperted yet.')
#         self.optimizers.append(self.optimizer_g)
#
#     def feed_data(self, data):
#         self.lq = data['lq'].to(self.device)
#         if 'gt' in data:
#             self.gt = data['gt'].to(self.device)
#
#     def optimize_parameters(self, current_iter):
#         self.optimizer_g.zero_grad()
#         self.output = self.net_g(self.lq)
#
#         l_total = 0
#         loss_dict = OrderedDict()
#         # pixel loss
#         if self.cri_pix:
#             l_pix = self.cri_pix(self.output, self.gt)
#             l_total += l_pix
#             loss_dict['l_pix'] = l_pix
#         # perceptual loss
#         if self.cri_perceptual:
#             l_percep, l_style = self.cri_perceptual(self.output, self.gt)
#             if l_percep is not None:
#                 l_total += l_percep
#                 loss_dict['l_percep'] = l_percep
#             if l_style is not None:
#                 l_total += l_style
#                 loss_dict['l_style'] = l_style
#
#         l_total.backward()
#         self.optimizer_g.step()
#
#         self.log_dict = self.reduce_loss_dict(loss_dict)
#
#     def test(self):
#         self.net_g.eval()
#         x = self.lq.squeeze(0).permute(1, 2, 0)
#         sr = []
#         with torch.no_grad():  # 在这里增加测试模型融合
#             for rot in range(0, 4):
#                 for flip in [False, True]:
#                     _x = x.flip([1]) if flip else x
#                     _x = _x.rot90(rot)
#                     out = self.net_g(_x.permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
#                     out = out.rot90(4 - rot)
#                     out = out.flip([1]) if flip else out
#                     sr.append(out)
#             self.output = torch.stack(sr).mean(0).permute(2, 0, 1)
#         self.net_g.train()
#
#     def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
#         logger = get_root_logger()
#         logger.info('Only support single GPU validation.')
#         self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
#
#     def nondist_validation(self, dataloader, current_iter, tb_logger,
#                            save_img):
#         dataset_name = dataloader.dataset.opt['name']
#         with_metrics = self.opt['val'].get('metrics') is not None
#         if with_metrics:
#             self.metric_results = {
#                 metric: 0
#                 for metric in self.opt['val']['metrics'].keys()
#             }
#         pbar = tqdm(total=len(dataloader), unit='image')
#
#         for idx, val_data in enumerate(dataloader):
#             img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
#             self.feed_data(val_data)
#             self.test()
#
#             visuals = self.get_current_visuals()
#             sr_img = tensor2img([visuals['result']])
#             if 'gt' in visuals:
#                 gt_img = tensor2img([visuals['gt']])
#                 del self.gt
#
#             # tentative for out of GPU memory
#             del self.lq
#             del self.output
#             torch.cuda.empty_cache()
#
#             if save_img:
#                 if self.opt['is_train']:
#                     save_img_path = osp.join(self.opt['path']['visualization'],
#                                              img_name,
#                                              f'{img_name}_{current_iter}.png')
#                 else:
#                     if self.opt['val']['suffix']:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["val"]["suffix"]}.png')
#                     else:
#                         save_img_path = osp.join(
#                             self.opt['path']['visualization'], dataset_name,
#                             f'{img_name}_{self.opt["name"]}.png')
#                 imwrite(sr_img, save_img_path)
#
#             if with_metrics:
#                 # calculate metrics
#                 opt_metric = deepcopy(self.opt['val']['metrics'])
#                 for name, opt_ in opt_metric.items():
#                     metric_type = opt_.pop('type')
#                     self.metric_results[name] += getattr(
#                         metric_module, metric_type)(sr_img, gt_img, **opt_)
#             pbar.update(1)
#             pbar.set_description(f'Test {img_name}')
#         pbar.close()
#
#         if with_metrics:
#             for metric in self.metric_results.keys():
#                 self.metric_results[metric] /= (idx + 1)
#
#             self._log_validation_metric_values(current_iter, dataset_name,
#                                                tb_logger)
#
#     def _log_validation_metric_values(self, current_iter, dataset_name,
#                                       tb_logger):
#         log_str = f'Validation {dataset_name}\n'
#         for metric, value in self.metric_results.items():
#             log_str += f'\t # {metric}: {value:.4f}\n'
#         logger = get_root_logger()
#         logger.info(log_str)
#         if tb_logger:
#             for metric, value in self.metric_results.items():
#                 tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)
#
#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict['lq'] = self.lq.detach().cpu()
#         out_dict['result'] = self.output.detach().cpu()
#         if hasattr(self, 'gt'):
#             out_dict['gt'] = self.gt.detach().cpu()
#         return out_dict
#
#     def save(self, epoch, current_iter):
#         self.save_network(self.net_g, 'net_g', current_iter)
#         self.save_training_state(epoch, current_iter)
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
import cv2

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class GaussianConv(nn.Module):
    def __init__(self, kernel_size=5, channels=3, sigma=2.0):
        super(GaussianConv, self).__init__()
        kernel = self.gauss_kernel(kernel_size, sigma)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(kernel, requires_grad=False).cuda()
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2, groups=self.channels)

    def gauss_kernel(self, size=5, sigma=2.0):
        grid = cv2.getGaussianKernel(size, sigma)
        kernel = grid * grid.T
        return kernel


class LaplacianPyramid(nn.Module):
    def __init__(self, max_level=5):
        super(LaplacianPyramid, self).__init__()
        self.gaussian_conv = GaussianConv()
        self.max_level = max_level

    def forward(self, X):
        t_pyr = []
        current = X
        for level in range(self.max_level - 1):
            t_guass = self.gaussian_conv(current)
            t_diff = current - t_guass
            t_pyr.append(t_diff)
            current = F.avg_pool2d(t_guass, 2)
        t_pyr.append(current)

        return t_pyr


class LaplacianLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='sum', max_level=5):
        super(LaplacianLoss, self).__init__()

        self.criterion = nn.L1Loss(reduction=reduction)
        self.lap = LaplacianPyramid(max_level=max_level)
        self.loss_weight = loss_weight

    def forward(self, x, y):
        x_lap, y_lap = self.lap(x), self.lap(y)
        diff_levels = [self.criterion(a, b) for a, b in zip(x_lap, y_lap)]
        return self.loss_weight * sum(2 ** (j - 1) * diff_levels[j] for j in range(len(diff_levels)))


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        # self.net_g = define_network(deepcopy(opt['network_g']))
        # self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)
        for i in range(6):
            setattr(self, 'net_g_' + str(i), self.model_to_device(define_network(deepcopy(opt['network_g']))))
        self.load_network(self.net_g_0,
                          '/home/rpf/tgp/BasicSR/experiments/WITH_MPR_PRE_256_l2_C/models/net_g_5000.pth',
                          self.opt['path'].get('strict_load_g', True))
        self.load_network(self.net_g_1,
                          '/home/rpf/tgp/BasicSR/experiments/WITH_MPR_PRE_256_l2_C/models/net_g_10000.pth',
                          self.opt['path'].get('strict_load_g', True))
        self.load_network(self.net_g_2,
                          '/home/rpf/tgp/BasicSR/experiments/WITH_MPR_PRE_256_l2_C/models/net_g_15000.pth',
                          self.opt['path'].get('strict_load_g', True))
        self.load_network(self.net_g_3,
                          '/home/rpf/tgp/BasicSR/experiments/WITH_MPR_PRE_256_l2_C/models/net_g_20000.pth',
                          self.opt['path'].get('strict_load_g', True))
        # self.load_network(self.net_g_6, load_path, self.opt['path'].get('strict_load_g', True))
        # self.load_network(self.net_g_7, load_path, self.opt['path'].get('strict_load_g', True))

        self.criterion_edge = EdgeLoss()
        self.laploss = LaplacianLoss()
        # load pretrained models
        # load_path = self.opt['path'].get('pretrain_network_g', None)
        # if load_path is not None:
        #     self.load_network(self.net_g, load_path,
        #                       self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params,
                                                **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.bic = data['bic'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def getdwt(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_HL, x_LH, x_HH), 1)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.lr_output, self.output = self.net_g(self.lq)  # 这里返回bic lr和sr,然后增加loss
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            pix_loss = self.cri_pix(self.output, self.gt) + self.cri_pix(self.lr_output, self.bic)
            hf_loss = self.cri_pix(self.getdwt(self.lr_output), self.getdwt(
                self.bic))  # + self.cri_pix(self.getdwt(self.output), self.getdwt(self.gt))
            # edge_loss = self.criterion_edge(self.output, self.gt)  # + self.criterion_edge(self.lr_output, self.bic)
            lap_loss = self.laploss(self.output, self.gt)
            l_pix = pix_loss + 10 * hf_loss + lap_loss * 1e-5
            # print('pix_loss,hf_loss,lap_loss', pix_loss.item(), hf_loss.item(), lap_loss.item())
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.lr_output, self.output = self.net_g(self.lq)
        self.net_g.train()

    def back_projection(self, iter, sr, lr):  # torch.Size([720, 1280, 3]) torch.Size([180, 320, 3]
        sr = sr.permute(2, 0, 1).unsqueeze(0)  # 超分结果
        lr = lr.permute(2, 0, 1).unsqueeze(0)  # 模糊小图
        for i in range(iter):
            # bic
            lr_bic = F.interpolate(sr, size=None, scale_factor=(0.25, 0.25), mode='bicubic', align_corners=False)
            sr = sr + self.net_g((lr - lr_bic).clone())[1]
            # sr = sr + F.interpolate(lr - lr_bic, size=None, scale_factor=(4, 4), mode='bicubic', align_corners=False)
        return sr.squeeze(0).permute(1, 2, 0)

    def final_test(self):  # 除了自相似集成，加入BP
        for i in range(4):
            getattr(self, 'net_g_' + str(i)).eval()
        x = self.lq.squeeze(0).permute(1, 2, 0)
        sr = []
        with torch.no_grad():  # 在这里增加测试模型融合
            for rot in range(0, 4):
                for flip in [False, True]:
                    _x = x.flip([1]) if flip else x
                    _x = _x.rot90(rot)
                    out_sr = self.net_g_0(_x.permute(2, 0, 1).unsqueeze(0).clone())
                    out_sr = out_sr.squeeze(0).permute(1, 2, 0)
                    out_sr = out_sr.rot90(4 - rot)
                    out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
                    sr.append(out_sr)

            for rot in range(0, 4):
                for flip in [False, True]:
                    _x = x.flip([1]) if flip else x
                    _x = _x.rot90(rot)
                    out_sr = self.net_g_1(_x.permute(2, 0, 1).unsqueeze(0).clone())
                    out_sr = out_sr.squeeze(0).permute(1, 2, 0)
                    out_sr = out_sr.rot90(4 - rot)
                    out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
                    sr.append(out_sr)

            for rot in range(0, 4):
                for flip in [False, True]:
                    _x = x.flip([1]) if flip else x
                    _x = _x.rot90(rot)
                    out_sr = self.net_g_2(_x.permute(2, 0, 1).unsqueeze(0).clone())
                    out_sr = out_sr.squeeze(0).permute(1, 2, 0)
                    out_sr = out_sr.rot90(4 - rot)
                    out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
                    sr.append(out_sr)

            for rot in range(0, 4):
                for flip in [False, True]:
                    _x = x.flip([1]) if flip else x
                    _x = _x.rot90(rot)
                    out_sr = self.net_g_3(_x.permute(2, 0, 1).unsqueeze(0).clone())
                    out_sr = out_sr.squeeze(0).permute(1, 2, 0)
                    out_sr = out_sr.rot90(4 - rot)
                    out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
                    sr.append(out_sr)

            #for rot in range(0, 4):
            #    for flip in [False, True]:
            #        _x = x.flip([1]) if flip else x
            #        _x = _x.rot90(rot)
            #        out_lr, out_sr = self.net_g_4(_x.permute(2, 0, 1).unsqueeze(0).clone())
            #        out_sr = out_sr.squeeze(0).permute(1, 2, 0)
            #        out_sr = out_sr.rot90(4 - rot)
            #        out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
            #        sr.append(out_sr)

            #for rot in range(0, 4):
            #    for flip in [False, True]:
            #        _x = x.flip([1]) if flip else x
            #        _x = _x.rot90(rot)
            #        out_lr, out_sr = self.net_g_5(_x.permute(2, 0, 1).unsqueeze(0).clone())
            #        out_sr = out_sr.squeeze(0).permute(1, 2, 0)
            #        out_sr = out_sr.rot90(4 - rot)
            #        out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
            #        sr.append(out_sr)

            # for rot in range(0, 4):
            #     for flip in [False, True]:
            #         _x = x.flip([1]) if flip else x
            #         _x = _x.rot90(rot)
            #         out_lr, out_sr = self.net_g_6(_x.permute(2, 0, 1).unsqueeze(0).clone())
            #         out_sr = out_sr.squeeze(0).permute(1, 2, 0)
            #         out_sr = out_sr.rot90(4 - rot)
            #         out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
            #         sr.append(out_sr)
            #
            # for rot in range(0, 4):
            #     for flip in [False, True]:
            #         _x = x.flip([1]) if flip else x
            #         _x = _x.rot90(rot)
            #         out_lr, out_sr = self.net_g_7(_x.permute(2, 0, 1).unsqueeze(0).clone())
            #         out_sr = out_sr.squeeze(0).permute(1, 2, 0)
            #         out_sr = out_sr.rot90(4 - rot)
            #         out_sr = out_sr.flip([1]) if flip else out_sr  # 720, 1280, 3
            #         sr.append(out_sr)
            self.output = torch.stack(sr).mean(0).permute(2, 0, 1).clamp(0, 1)

        for i in range(4):
            getattr(self, 'net_g_' + str(i)).train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.info('Only support single GPU validation.')
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.final_test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'], dataset_name,
                            f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                for name, opt_ in opt_metric.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(
                        metric_module, metric_type)(sr_img, gt_img, **opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
