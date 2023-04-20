import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
import re
import pandas as pd
from tqdm.auto import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    
    argparser = ArgumentParser()
    argparser.add_argument('config', type=str, help='A config yaml file, like denoise_example.yml')

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    exp_dir = config.expdir_path
    manifest_paths = config.manifest_path
    asr_config_path = config.asr.config_path
    asr_ckpt_path = config.asr.checkpoint_path
   
    device = torch.device('cuda:0')

    for manifest_path in manifest_paths.split(','):

        df = pd.read_json(manifest_path, lines=True)

        asr_config = OmegaConf.load(asr_config_path)
        asr_config.trainer.progress_bar_refresh_rate = 10

        # trainer = pl.Trainer(**asr_config.trainer)
        asr_model = nemo_asr.models.EncDecCTCSpecModelBPE(cfg=asr_config.model, trainer=None)
        state_dict = torch.load(asr_ckpt_path, map_location=device)#['state_dict']
       # __import__('ipdb').set_trace()
        asr_model.load_state_dict(
           state_dict)
        asr_model.to(torch.device('cuda:0'))
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        transcripts = asr_model.transcribe(df['audio_filepath'])
        df['predictions'] = transcripts

        df['predictions'] = df['predictions'].fillna('')

        pattern = r'[^a-zA-Z\ ]'
        df['text'] = df['text'].apply(lambda x: x.replace("'", " "))
        df['text'] = df['text'].apply(lambda x: re.sub(pattern, '', x.strip().lower()))
        #__import__('ipdb').set_trace()

        df.to_csv(exp_dir + manifest_path.split('/')[-1].split('.')[0]  + '_asr_eval.csv', encoding='utf-8', index=True )

    # exp_manager(trainer, config.get("exp_manager", None))
