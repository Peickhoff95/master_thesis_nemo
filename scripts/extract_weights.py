import torch
from nemo.collections.asr.models import EncDecCTCModel
from omegaconf import OmegaConf

def extract_pt_from_nemo(in_path, out_path):
    model = EncDecCTCModel.restore_from(restore_path=in_path)

    # Save state_dict
    torch.save(model.state_dict(), out_path)

def extract_conf_from_nemo(in_path, out_path):
    model = EncDecCTCModel.restore_from(restore_path=in_path, return_config=True)

    # Save config
    with open(out_path, "w") as outfile:
        OmegaConf.save(config=model, f=outfile)
