import numpy as np
import torch
import matplotlib.pyplot as plt

def get_nrow_ncol(n):
    for i in range(int(n**0.5)+1, 0, -1):
        if n % i == 0:
            nrows = i
            break
    ncols = n // nrows
    return (nrows, ncols)

def show(imgs: np.array,
         cmap='seismic',
         show=True, save=True,
         nrows=None, ncols=None,
         fig_title=None, ax_titles=None,
         save_path='img.jpg', flatten_color_ch=False):
    """
    imgs (np.array)
        Should have 4 dimensions (n_images, n_channels, height, width)
    """
    
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()
    
    if len(imgs.shape) == 3:
        imgs = np.expand_dims(imgs, 0)
    assert len(imgs.shape)==4, "'imgs' should be np.array with 4 dimensions (n_images, n_channels, height, width)"
    assert imgs.shape[1] in [1, 3], "Dimension 1 of 'imgs' should be 0 or 3 (n_channels)"
    
    if imgs.shape[1] == 3:
        if flatten_color_ch:
            imgs = np.mean(imgs, 1)
            imgs = np.expand_dims(imgs, 1)
    
    imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    
    n_imgs = imgs.shape[0]
    if nrows==None and ncols==None:
        nrows, ncols = get_nrow_ncol(n_imgs)
    
    if nrows>1 and ncols>1:
        fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), gridspec_kw={'wspace': 0.1})
        for i, img in enumerate(imgs):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if (len(img.shape)==3) and (img.shape[0]==3):
                img = np.transpose(img, (1,2,0))
            if (len(img.shape)==3) and (img.shape[0]==1):
                img = img[0]
            ax[i//ncols, i%ncols].set_axis_off()
            ax[i//ncols, i%ncols].imshow(img, cmap=cmap)
            if ax_titles is not None:
                ax[i//ncols, i%ncols].set_title(ax_titles[i])
    elif nrows>1 or ncols>1:
        fig, ax = plt.subplots(nrows,ncols, figsize=(5*ncols,5*nrows), gridspec_kw={'wspace': 0.1})
        for i, img in enumerate(imgs):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if (len(img.shape)==3) and (img.shape[0]==3):
                img = np.transpose(img, (1,2,0))
            if (len(img.shape)==3) and (img.shape[0]==1):
                img = img[0]
            ax[i].set_axis_off()
            ax[i].imshow(img, cmap=cmap)
            if ax_titles is not None:
                ax[i].set_title(ax_titles[i])
    else:
        fig, ax = plt.subplots(1,1, figsize=(5,5), gridspec_kw={'wspace': 0.1})
        img = imgs[0]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if (len(img.shape)==3) and (img.shape[0]==3):
            img = np.transpose(img, (1,2,0))
        if (len(img.shape)==3) and (img.shape[0]==1):
            img = img[0]
        ax.set_axis_off()
        ax.imshow(img, cmap=cmap)
        if ax_titles is not None:
            if isinstance(ax_titles, str):
                ax_titles = [ax_titles]
            ax.set_title(ax_titles[0])
    fig.suptitle(fig_title)
    plt.subplots_adjust(top=0.75)
    # plt.tight_layout(rect=[0, 0, 5, 2])
    if show:
        plt.show()
    if save:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

def clear_color(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def compute_beta(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    b = beta.index_select(0, t + 1).view(-1, 1, 1, 1)
    return b

def batchfy(tensor, batch_size):
    n = len(tensor)
    num_batches = n // batch_size + 1
    return tensor.chunk(num_batches, dim=0)


def _renormalize_gradient(grad, eps, small_const=1e-22):
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    #print('grad norm is', grad_norm)
    grad_norm = torch.where(grad_norm < small_const, grad_norm+small_const, grad_norm)
    grad /= grad_norm
    grad *= eps.view(grad.shape[0], -1).norm(p=2, dim=1).view(grad.shape[0], 1, 1, 1)
    return grad


def compute_lp_gradient(diff, p, small_const=1e-12):
    if p < 1:
        grad_temp = (p * (diff.abs() + small_const) ** (p - 1)) * diff.sign()
    else:
        grad_temp = (p * diff.abs() ** (p - 1)) * diff.sign()
    return grad_temp