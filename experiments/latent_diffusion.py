import argparse
import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate
from omegaconf import OmegaConf
import hydra
from PIL import Image
from tqdm import tqdm
from experiments.data_utils import SirenAndOriginalDataset
from nfn.common import network_spec_from_wsfeat, WeightSpaceFeatures
from experiments.siren_utils import unprocess_img_arr
from experiments.diffusion_utils import decode_latents, EmbeddingData

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper
from torchdyn.core import NeuralODE
def train(cfg):
    #only loading the original SIRENS because we have to to load the NFNet...
    dset = SirenAndOriginalDataset(cfg.dset.siren_path, "randinit_smaller", cfg.dset.data_path)
    spec = network_spec_from_wsfeat(WeightSpaceFeatures(*next(iter(DataLoader(dset, batch_size=1)))[0]).to("cpu"), set_all_dims=True)
        
    nfnet = hydra.utils.instantiate(cfg.model, spec, dset_data_type=dset.data_type, compile=False).to("cuda")
    nfnet.load_state_dict(torch.load(os.path.join(args.rundir, "best_nfnet.pt")))

    #freeze weights
    nfnet.eval()
    for param in nfnet.parameters():
        param.requires_grad = False
    
    device = 'cuda'

    #load the SIREN embeddings (a.k.a the latents fron INR2Array)
    dataset = EmbeddingData(args.latents_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    latents, class_labels = next(iter(loader))
    latents = latents.to(device)

    #exclude batch dim
    latent_shape = latents.shape[1:]

    print("LATENT SHAPE", latent_shape)

    # setup denoising UNet and ODE solver
    sigma = 0.0
    model = UNetModelWrapper(
        dim=(1,64,64), num_channels=32, num_res_blocks=2, num_classes=10, class_cond=True
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    FM = ConditionalFlowMatcher(sigma=sigma)
    # ode_solver = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    EPOCHS = 100
    for epoch in range(EPOCHS):
        for i, (latents, class_labels) in enumerate(loader):
            optimizer.zero_grad()
            x1 = latents.to(device)
            y = class_labels.to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            B, C, H, W = xt.shape  # [4, 1, 16, 256]

            #The UNet can only deal with square tensors, so we need to reshape our latent from [1,16,256] -> [1,64,64]
            xt = xt.view(B, C, 64, 64)
            ut = ut.view(B, C, 64, 64)
            vt = model(t, xt, y)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
            
    
    

    # img_tensors = decode_latents(nfnet, latents)
    # save_image(img_tensors, "test_decode.png")


def main(args):
    import os
    print(os.path.abspath(args.latents_path))
    cfg = OmegaConf.load(os.path.join(args.rundir, ".hydra/config.yaml"))
    train(cfg)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rundir", type=str, required=True)
    parser.add_argument("--latents_path", type=str, required=True)
    args = parser.parse_args()
    main(args)

#python -m experiments.latent_diffusion --rundir "./outputs/2025-01-12/19-49-16" --latents_path "/users/ksaripal/data/ksaripal/nft-embeddings/mnist-embeddings.pt"