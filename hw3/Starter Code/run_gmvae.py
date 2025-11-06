# Copyright (c) 2018 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
import utils as ut
from gmvae import GMVAE
from train import train
from pprint import pprint
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=10,    help="Number of latent dimensions")
parser.add_argument('--k',         type=int, default=500,   help="Number mixture components in MoG prior")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',  'gmvae'),
    ('z={:02d}',  args.z),
    ('k={:03d}',  args.k),
    ('run={:04d}', args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
gmvae = GMVAE(z_dim=args.z, k=args.k, name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=gmvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound(gmvae, labeled_subset, run_iwae=args.train == 2)

    from torchvision.utils import save_image

    # 确保模型在评估模式
    gmvae.eval()

    # 任务 3：可视化 200 个样本
    print("Saving 200 generated samples to 'gmvae_samples.png'...")

    # 1. 从先验 p(z) 采样，并通过解码器生成图像
    # 这个函数会为你完成 "从 p(z) 采样" (sample_z) 
    # 和 "通过解码器" (compute_sigmoid_given) 两个步骤
    with torch.no_grad(): # 我们不需要计算梯度
        generated_images_sig = gmvae.sample_sigmoid(batch=200)

    # 2. 把 (200, 784) 的张量变形为 (200, 1, 28, 28)
    # 这是 save_image 需要的格式 (batch, channels, height, width)
    generated_images = generated_images_sig.reshape(200, 1, 28, 28)

    # 3. 保存成一个 10x20 的网格
    # "nrow=20" 告诉 save_image 每行放 20 张图
    # 因为我们有 200 张，它会自动生成 10 行
    save_image(generated_images.cpu(),
            'gmvae_samples.png',
            nrow=20)

    print("Visualization saved!")

else:
    ut.load_model_by_name(gmvae, global_step=args.iter_max, device=device)
    ut.evaluate_lower_bound(gmvae, labeled_subset, run_iwae=True)

    from torchvision.utils import save_image
    gmvae.eval()
    print("Saving 200 generated samples from loaded model...")
    with torch.no_grad():
        generated_images_sig = gmvae.sample_sigmoid(batch=200)
    generated_images = generated_images_sig.reshape(200, 1, 28, 28)
    save_image(generated_images.cpu(),
               'gmvae_samples_from_loaded.png',
               nrow=20)
    print("Visualization saved!")


