#%%
import os
import csv
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from pathlib import Path
from blended_diffusion_custom.guided_diffusion.guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from utils import normalize_np, clear_color, compute_alpha, compute_beta, batchfy, _renormalize_gradient, compute_lp_gradient

device = torch.device(type='cuda', index=0)

def get_args_parser():
    parser = argparse.ArgumentParser('freemcg', add_help=False)
    parser.add_argument('--weights', default='torchvision.models.ResNet50_Weights.IMAGENET1K_V2')
    parser.add_argument('--arch', default='torchvision.models.resnet50')
    parser.add_argument('--diffusion_ckpt_path', default='diffusion_ckpts/openai-guided-diffusion/256x256_diffusion_uncond.pt')
    parser.add_argument('--imagenet_dir', default='/home/wonjun/data/ILSVRC2012/val')
    parser.add_argument('--gpu_id', default=0)
    parser.add_argument('--gpu', default=[0])
    parser.add_argument('--device', default=torch.device(type='cuda', index=0))
    parser.add_argument('--n_particles', type=int, default=100)
    parser.add_argument('--src_category', default='cheetah')
    parser.add_argument('--src_img_ind', type=int, default=2)
    parser.add_argument('--dst_category', default='tiger')
    parser.add_argument('--n_diffusion_steps', type=int, default=100)
    parser.add_argument('--step_size', default=0.2, type=float)
    parser.add_argument('--step_size_prox', default=0.02, type=float)
    parser.add_argument('--strength', default=0.4, type=float)
    parser.add_argument('--eta', default=1.0, type=float)
    parser.add_argument('--save_root', default='outputs', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_classifier_logits(x, classifier, model_transforms):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if len(x.shape) == 3:
        x = np.expand_dims(x, 0)
    x = x.transpose(0,2,3,1)
    
    x -= x.min()
    x /= x.max()
    x = (x*255).astype(np.uint8)
    
    # TODO: remove this loop? doesn't seem necessary
    preds = []
    for i in range(x.shape[0]):
        im = Image.fromarray(x[i])
        im = model_transforms(im)
        pred = classifier(im.unsqueeze(0).to(device))
        preds.append(pred)
    pred = torch.cat(preds, dim=0)
    return pred

@torch.no_grad()
def generate_counterfactual(
    x,
    classifier,
    classifier_transforms,
    diffusion_model,
    diffusion_utils,
    dst_category_ind, # destination category index
    src_img_ind,
    n_particles=100,
    n_diffusion_steps=100,
    strength=0.4,
    eta=0.0,
    step_size=0.2,
    step_size_prox=0.02,
    save_path="counterfactual.png",
):
    softmax = nn.Softmax()
    
    x_orig = torch.tensor(x).to(device)
    x_orig = T.Resize((256,256))(x_orig)
    x_orig -= x_orig.min()
    x_orig /= x_orig.max()
    x_orig = x_orig * 2 - 1 # range [-1, 1]
    
    fx = get_classifier_logits(x_orig, classifier, classifier_transforms)
    
    ec = torch.zeros_like(fx)
    ec[:, dst_category_ind] = 1
    
    start_step = int(1000 * strength)
    skip = start_step // n_diffusion_steps
    
    # generate time schedule
    times = range(0, start_step, skip)
    times_next = [-1] + list(times[:-1])
    times_pair = zip(reversed(times), reversed(times_next))
    
    # x_t^k \sim q(x_t|x_0)
    betas = torch.tensor(diffusion_utils.betas).to(device)
    noise = torch.randn(n_particles, 3, 256, 256).to(device)
    start_t = (torch.ones(1) * start_step).to(device)
    at_start = compute_alpha(betas, start_t.long())
    x = at_start.sqrt() * x_orig + (1 - at_start).sqrt() * noise
    xs = [x]
    
    # reverse diffusion
    pbar = tqdm(times_pair, total=len(times))
    logit_list = []
    for i, j in pbar:
        t = (torch.ones(1) * i).to(x.device)
        next_t = (torch.ones(1) * j).to(x.device)
        
        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        
        xt = xs[-1].to(device).float()
        
        # 0. NFE.Batchfy needed to fit into GPU memory
        xt_batch = batchfy(xt, 10)
        et_agg = list()
        for _, xt_batch_sing in enumerate(xt_batch):
            t = torch.ones(xt_batch_sing.shape[0], device=device) * i
            et_sing = diffusion_model(xt_batch_sing, t)
            et_agg.append(et_sing)
        et = torch.cat(et_agg, dim=0)
        et = et[:, :et.size(1)//2]
        
        # 1. Tweedie-denoising
        x0_t = (xt - et * (1-at).sqrt()) / at.sqrt()
        x0_t = torch.clip(x0_t, -1, 1)
        x0_t_bar = x0_t.mean(dim=0, keepdim=True)
        
        # 2. Calculate FreeMCG (i.e. approximation of gradient)
        fx0_t = get_classifier_logits(x0_t, classifier, classifier_transforms)
        px_j = softmax(fx0_t)
        ec = torch.zeros_like(px_j)
        ec[:, dst_category_ind] = 1
        ecmp_j = ec - px_j
        
        f_bar = fx0_t.mean(dim=0, keepdim=True)
        
        # log the current average logit value for the destination class (class of counterfactual)
        pbar.set_description(f"Logit value: {fx0_t[0, dst_category_ind].item():.2f}")
        logit_list.append(fx0_t[:, dst_category_ind].mean().item())
        
        fmfbar = fx0_t - f_bar
        xmxbar = x0_t - x0_t_bar
        
        weights = torch.sum(fmfbar * ecmp_j, dim=1)
        weights = weights.view(-1, 1, 1, 1)
        freemcg = xmxbar * weights
        
        freemcg = _renormalize_gradient(freemcg, et)
        freemcg = freemcg * step_size
        
        lp_grad = x0_t - x_orig
        lp_grad = compute_lp_gradient(lp_grad, 1.0)
        lp_grad = _renormalize_gradient(lp_grad, et)
        lp_grad = lp_grad * step_size_prox
        
        freemcg -= lp_grad
        c1 = (
            eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        
        grad_coeff = at.sqrt() * at_next.sqrt()

        if j != 0:
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et + freemcg * grad_coeff
        else:
            xt_next = x0_t
        xs.append(xt_next.to('cpu'))        
        
    # plt.plot(logit_list)
    # plt.savefig(save_path / f"logit_list{src_img_ind}.png")
    # with open(save_path / f"logit_list{src_img_ind}.csv", "w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(logit_list)
    
    return xs, logit_list

def main():
    parser = get_args_parser()
    hps = parser.parse_args()
    print("---------------------------------------------------")
    print(f"{hps.src_category}[{hps.src_img_ind}] to {hps.dst_category} (prox {hps.step_size_prox})")
    print("---------------------------------------------------")
    
    set_seeds(hps.seed)
    
    hps.device_ids = [int(hps.gpu[0])]
    device = torch.device('cuda:'+str(hps.gpu[0]))
    hps.device = device
    
    # classifier = torchvision.models.resnet50(weights=eval(hps.weights))
    classifier = eval(hps.arch)(weights=eval(hps.weights))
    classifier.eval()
    classifier.to(device)
    
    classifier_transforms = eval(hps.weights).transforms()
    
    categories = eval(hps.weights).meta['categories']
    folders = sorted(os.listdir(hps.imagenet_dir))
    category_to_folder = {cat:folder for cat, folder in zip(categories, folders)}
    
    imgs_dir = os.path.join(hps.imagenet_dir, category_to_folder[hps.src_category])
    imgs_list = sorted(os.listdir(imgs_dir))
    img_name = imgs_list[int(hps.src_img_ind)]
    img_path = os.path.join(imgs_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    
    # img_t = eval(hps.weights).transforms()(img)
    t = T.Compose([
        T.Resize((224,224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = t(img)
    
    x = np.expand_dims(img_t, 0)
    save_path = Path(f"./{hps.save_root}/{hps.src_category}_to_{hps.dst_category}/n_diffusion_steps{hps.n_diffusion_steps}/step_size_{hps.step_size}/step_size_prox_{hps.step_size_prox}/strength_{hps.strength}/eta{hps.eta}")
    save_path.mkdir(exist_ok=True, parents=True)
    
    # y_batch = np.array([categories.index(hps.src_category)] * len(x))
    # x_sequence = []
    # fx_sequence = []
    
    # diffusion_ckpt_path = os.path.join(
    #     'diffusion_ckpts', 'openai-guided-diffusion', '256x256_diffusion_uncond.pt'
    # )
    diffusion_ckpt_path = hps.diffusion_ckpt_path
    
    # Load diffusion model
    diffusion_config={
        'image_size': 256,
        'num_channels' : 256, 
        'num_res_blocks': 2,
        'resblock_updown': True,
        'num_heads': 4,
        'num_heads_upsample': -1,
        'num_head_channels': 64,
        'attention_resolutions' : '32, 16, 8',
        'channel_mult': '',
        'dropout': 0.0,
        'class_cond': False,
        'use_checkpoint': False,
        'use_scale_shift_norm': True,
        'use_fp16': True,
        'use_new_attention_order': False,
        'learn_sigma': True,
        'diffusion_steps': 1000,
        'noise_schedule': 'linear',
        'timestep_respacing': "1000", 
        'use_kl': False,
        'predict_xstart': False,
        'rescale_timesteps': True,
        'rescale_learned_sigmas': False
    }
    diffusion_model, diffusion_utils = create_model_and_diffusion(**diffusion_config)
    diffusion_model.load_state_dict(
        torch.load(diffusion_ckpt_path, map_location='cpu')
    )
    diffusion_model.requires_grad_(False).eval().to(device)
    for name, param in diffusion_model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if diffusion_config['use_fp16']:
        diffusion_model.convert_to_fp16()
    
    cf, logit_list = generate_counterfactual(
        x=x, 
        classifier=classifier, 
        classifier_transforms=classifier_transforms,
        diffusion_model=diffusion_model,
        diffusion_utils=diffusion_utils,
        dst_category_ind=categories.index(hps.dst_category), 
        src_img_ind=hps.src_img_ind,
        n_particles=int(hps.n_particles),
        n_diffusion_steps = int(hps.n_diffusion_steps),
        strength=float(hps.strength),
        eta=float(hps.eta),
        step_size=float(hps.step_size), 
        step_size_prox=float(hps.step_size_prox), 
        save_path=save_path,
    )
    cf_mean = cf[-1].mean(dim=0)

    x = np.transpose(x[0], (1,2,0))
    x = normalize_np(x)
    plt.imsave(str(save_path / f"orig{hps.src_img_ind}.jpg"), x)
    plt.imsave(str(save_path / f"cntf{hps.src_img_ind}_seed{hps.seed}.jpg"), clear_color(cf_mean))
    plt.plot(logit_list)
    plt.savefig(save_path / f"logit_list{hps.src_img_ind}_seed{hps.seed}.png")
    
if __name__ == "__main__":
    main()
# %%
