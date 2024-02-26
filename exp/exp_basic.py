import os
import torch
import numpy as np
import logging


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    
    def get_logger(self, name):
        logger = logging.getLogger(name)
        # 创建一个handler，用于写入日志文件
        filename = f'{name}.log'
        fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
        # 再创建一个handler用于输出到控制台
        ch = logging.StreamHandler()
        # 定义输出格式(可以定义多个输出格式例formatter1，formatter2)
        formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        # 定义日志输出层级
        logger.setLevel(logging.DEBUG)
        # 定义控制台输出层级
        # logger.setLevel(logging.DEBUG)
        # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
        fh.setFormatter(formatter)
        # 为控制台操作符绑定格式（可以绑定多种格式例ch.setFormatter(formatter2)）
        ch.setFormatter(formatter)
        # 给logger对象绑定文件操作符
        logger.addHandler(fh)
        # 给logger对象绑定文件操作符
        logger.addHandler(ch)

        return logger
