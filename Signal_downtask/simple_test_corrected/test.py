
import argparse
from typing import Tuple
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
sys.path.append('/home/dingzhengyao/Work/PPG-ECG/Project/P2E_v2/Signal_downtask/LifeSense')
from memory_profiler import profile

import SingnalEncoder as SingnalEncoder
from dataset import get_dataset

def calculate_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    计算 TP, FP, TN, FN。
    
    参数:
    y_true -- 真实标签 (0 或 1)
    y_pred -- 预测概率
    threshold -- 用于将概率转换为标签的阈值，默认是 0.5
    
    返回:
    TP -- 真阳性
    FP -- 假阳性
    TN -- 真阴性
    FN -- 假阴性
    """
    # 根据阈值将预测的概率转为二分类标签
    y_pred_label = (y_pred >= threshold).astype(int)
    
    # 计算 TP, FP, TN, FN
    TP = np.sum((y_true == 1) & (y_pred_label == 1))
    FP = np.sum((y_true == 0) & (y_pred_label == 1))
    TN = np.sum((y_true == 0) & (y_pred_label == 0))
    FN = np.sum((y_true == 1) & (y_pred_label == 0))
    
    return TP, FP, TN, FN


def calculate_metrics(y_true, y_pred):
    # 计算基本指标
    TP, FP, TN, FN = calculate_confusion_matrix(y_true, y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0

    return sensitivity,specificity,accuracy,F1



@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    
    
    model.eval()
    pred = []
    real = []
    loss_fn = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(tqdm(data_loader, desc="Validation")):
        
        signal = batch['ppg'].to(device, dtype=torch.float32)
        signal = signal.unsqueeze(1).unsqueeze(1)
        target = batch['label'].to(device, dtype=torch.float32).unsqueeze(1)
        
        with torch.cuda.amp.autocast():
            _, out = model(signal)
            pred.append(out)
            real.append(target)

    pred = torch.cat(pred, dim=0)
    real = torch.cat(real, dim=0)
    
    pred = torch.sigmoid(pred)
    pred = pred.cpu().numpy()
    real = real.cpu().numpy()
    # calculate auc
    AUC = roc_auc_score(real, pred)
    sensitivity, specificity, accuracy, F1 = calculate_metrics(real, pred)

    return {
        'y_pred': pred,
        'y_true': real,
        'AUC': AUC,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Accuracy': accuracy,
        'F1': F1
    }

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子 {seed} 已经设置完成，确保实验可复现。")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    
    
    # PPG Model parameters
    
    parser.add_argument('--latent_dim', default=1, type=int, metavar='N',help='latent_dim')
    parser.add_argument('--ppg_model', default='vit_base_patchX', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ppg_pretrained_model',
                        default="checkpoint/checkpoint-4-AUC-0.966.pth",
                        type=str, metavar='MODEL', help='path of pretaained model')

    parser.add_argument('--ppg_input_channels', type=int, default=1, metavar='N',
                        help='ppginput_channels')
    parser.add_argument('--ppg_input_electrodes', type=int, default=1, metavar='N',
                        help='ppg input electrodes')
    parser.add_argument('--ppg_time_steps', type=int, default=2560, metavar='N',
                        help='ppg input length')
    parser.add_argument('--ppg_input_size', default=(1, 2560), type=Tuple,
                        help='ppg input size')
    parser.add_argument('--ppg_patch_height', type=int, default=1, metavar='N',
                        help='ppg patch height')
    parser.add_argument('--ppg_patch_width', type=int, default=128, metavar='N',
                        help='ppg patch width')
    parser.add_argument('--ppg_patch_size', default=(1, 128), type=Tuple,
                        help='ppg patch size')
    parser.add_argument('--ppg_globle_pool', default=False,type=str2bool, help='ppg_globle_pool')
    parser.add_argument('--ppg_drop_out', default=0.1, type=float)

    
    # Dataset parameters
    parser.add_argument('--label', default='427_31', type=str, help='label type')
    parser.add_argument('--dataset', default='MIMIC_AF', type=str, help='dataset type')
    parser.add_argument('--balance', default=True, type=str2bool, help='balance')

    parser.add_argument('--device', default='cuda:2',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', default=True, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    return parser


def main(args):
    
    args.patch_size = (args.ppg_patch_height, args.ppg_patch_width)
    device = torch.device(args.device)
    

    # fix the seed for reproducibility
    seed = args.seed
    set_random_seed(seed)
    cudnn.benchmark = True
    

    dataset_val = get_dataset(args, mode='test')
    print("Validation set size: ", len(dataset_val))
    
    model = SingnalEncoder.__dict__[args.ppg_model](
            img_size=args.ppg_input_size,
            patch_size=args.ppg_patch_size,
            in_chans=args.ppg_input_channels,
            num_classes=args.latent_dim,
            drop_rate=args.ppg_drop_out,
            args=args)
    
    ppg_checkpoint = torch.load(args.ppg_pretrained_model, map_location='cpu')
    ppg_checkpoint_model = ppg_checkpoint['model']
    msg = model.load_state_dict(ppg_checkpoint_model, strict=False)
    print('load pretrained model')
    print(msg)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_in_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    print(f"Total model size: {total_size_in_bytes / (1024 ** 2):.2f} MB")
    result = evaluate(data_loader_val, model, device, args=args)
    return result


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    
    result = main(args)
    y_pred = result['y_pred']
    print(f'AUC: {result["AUC"]:.4f}, Sensitivity: {result["Sensitivity"]:.4f}, Specificity: {result["Specificity"]:.4f}, Accuracy: {result["Accuracy"]:.4f}, F1: {result["F1"]:.4f}')
