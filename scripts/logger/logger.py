#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2025/02/13 15:40:54
@Author  :   jzj
@Email   :   jzj.123@foxmail.com
@Version :   1.0
@description    :   xxxxx
'''

import logging
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored

FORMAT_CONFIG = {
    # 'rl': {
    #     'train': [
    #         ('episode', 'E', 'int'), ('step', 'S', 'int'), 
    #         ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'), ('train/epsiode step', 'ES', 'int'),
    #         ('batch_reward', 'BR', 'float'), ('actor_loss', 'ALOSS', 'float'),
    #         ('critic_loss', 'CLOSS', 'float'), ('ae_loss', 'RLOSS', 'float'),
    #         ('batch_rho', 'RHO', 'float'),
    #         ('max_rat', 'MR', 'float'), ('env_id', 'EI', 'int')
    #     ],
    #     'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float')]
    # }
    'rl': {
        'train': [
            ('episode', 'Episode', 'int'), ('step', 'Step', 'int'), 
            ('duration', 'Time', 'time'), ('episode_reward', 'EpisodeReward', 'float'), ('train/epsiode step', 'EpsiodeStep', 'int'),
            ('batch_reward', 'BatchReward', 'float'), ('batch_reward_agent', 'BatchRewardAgent', 'float'), ('batch_reward_expert', 'BatchRewardExpert', 'float'), 
            ('actor_loss', 'ALOSS', 'float'), ('actor_target_entropy', 'ATENT', 'float'), ('actor_entropy', 'AENT', 'float'),
            ('alpha_loss', 'ALPHALOSS', 'float'), ('alpha_value', 'ALPHA', 'float'),
            ('actor_guidence_loss', 'GUIDENCE_LOSS', 'float'), 
            ('critic_loss', 'CLOSS', 'float'), ('critic_target_Q', 'target_Q', 'float'), ('critic_current_Q1', 'Q1', 'float'), ('critic_current_Q2', 'Q2', 'float'),
            ('critic_target_Q_crop', 'target_Q_crop', 'float'), ('critic_current_Q1_crop', 'Q1_crop', 'float'), ('critic_current_Q2_crop', 'Q2_crop', 'float'),
            ('encoder_loss', 'ELOSS', 'float'), ('encoder_loss_pixel', 'ELOSS_PIXEL', 'float'), ('encoder_loss_window', 'ELOSS_WINDOW', 'float'),
            ('batch_rho', 'RHO', 'float'),
            ('max_rat', 'MaxRat', 'float'), ('env_id', 'EnvId', 'int')
        ],
        'eval': [('step', 'Step', 'int'), ('episode_reward', 'EpisodeReward', 'float')]
    }
}

class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        color = 'yellow' if prefix == 'train' else 'green'
        prefix = colored(prefix, color)
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            formatted_value = self._format(disp_key, value, ty)
            formatted_value_colored = colored(formatted_value, color)
            pieces.append(formatted_value_colored)
        # pieces = ' | '.join(pieces)
        return '| %s |' % (' | '.join(pieces))
        # return colored(' | ', color) + colored(pieces, color) + ' ' + colored('|', color)

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        # self._dump_to_file(data)
        dump_str = self._dump_to_console(data, prefix)
        self._meters.clear()
        return dump_str


class Logger:
    
    def __init__(self, logger_name, log_dir, config="rl"):
        """
        创建logger, 包含一个Logger和一个SummaryWriter
        """
        # logging.basicConfig(level=logging.INFO,
        #                 format='%(asctime)s | %(levelname)s | %(filename)-10s:%(lineno)-3d - %(message)s',
        #                 datefmt='%Y-%m-%d %H:%M:%S.%f')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.remove_all_logfile(log_dir)
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # 创建handler，用于写入日志文件
        file_handler = logging.FileHandler(log_dir + '/logfile.log')
        file_handler.setLevel(logging.INFO)  # 设置handler级别

        # 创建handler，用于将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)-10s:%(lineno)-3d - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加handler到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self._logger = logger
        self._sw = SummaryWriter(log_dir)

        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )

    def info(self, msg, *args, **kwargs):
        stacklevel = kwargs.pop('stacklevel', 2)
        self._logger.info(msg, *args, **kwargs, stacklevel=stacklevel + 1)
    
    def warning(self, msg, *args, **kwargs):
        stacklevel = kwargs.pop('stacklevel', 2)
        self._logger.warning(msg, *args, **kwargs, stacklevel=stacklevel + 1)

    def error(self, msg, *args, **kwargs):
        stacklevel = kwargs.pop('stacklevel', 2)
        self._logger.error(msg, *args, **kwargs, stacklevel=stacklevel + 1)
    
    def add_scalar(self, key, value, step):
        # self._sw.add_scalar(key, value, step)
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._sw.add_scalar(key, value, step)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value)

    def add_histogram(self, key, histogram, step):
        # self._sw.add_histogram(self, key, histogram, step)
        # Not support yet.
        pass

    def add_param(self, key, histogram, step):
        # Not support yet.
        pass

    def dump(self, step):
        train_str = self._train_mg.dump(step, 'train')
        if train_str:
            self._logger.info(f"train briefing: \n {train_str}", stacklevel=3)

        eval_str = self._eval_mg.dump(step, 'eval')
        if eval_str:
            self._logger.info(f"eval briefing: {eval_str}", stacklevel=3)

    def remove_all_logfile(self, log_dir):
        # 检查目录是否存在
        if not os.path.exists(log_dir):
            return

        # 遍历目录中的所有文件和子目录
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                # 如果是文件，则删除
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # 如果是目录，则递归删除
                    shutil.rmtree(file_path)
            except Exception as e:
                raise e
            
    def log_dict(self, msg, info_dict):
        format_str = msg
        for key, item in info_dict:
            format_str += " | "
            format_str += self._format(key, item[0], item[1])

        self._logger.info(format_str, stacklevel=3)

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)