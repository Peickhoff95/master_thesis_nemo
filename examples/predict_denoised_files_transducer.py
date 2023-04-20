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

def normalize_transducer_output(text: str):
    pattern = r'[^a-zA-Z\ ]'
    text = [x.replace("'ve ", " have") for x in text]
    text = [x.replace("'", " ") for x in text]
    text = [re.sub(pattern, '', x.strip().lower()) for x in text]
    text = [x.replace("mister", "mr") for x in text]
        
    return text
        

if __name__ == '__main__':

    
    argparser = ArgumentParser()
    argparser.add_argument('config', type=str, help='A config yaml file, like denoise_example.yml')

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    exp_dir = config.expdir_path
    manifest_paths = config.manifest_path
    rec_config_path = config.reconstruction.config_path
    rec_ckpt_path = config.reconstruction.checkpoint_path
   # asr_config_path = config.asr.config_path
   # asr_ckpt_path = config.asr.checkpoint_path
   
    device = torch.device('cuda:0')

    for manifest_path in manifest_paths.split(','):

        df = pd.read_json(manifest_path, lines=True)

        rec_config = OmegaConf.load(rec_config_path)
        rec_model = nemo_asr.models.ReconstructionModel(cfg=rec_config.model, trainer=None)
        chkpt = torch.load(rec_ckpt_path, map_location=device)

        rec_model.load_state_dict(chkpt['state_dict'])
        rec_model.to(device)
        denoised_specs = rec_model.reconstruct(df['input'])

        del rec_model

       # asr_config = OmegaConf.load(asr_config_path)
       # asr_config.trainer.progress_bar_refresh_rate = 10

        # trainer = pl.Trainer(**asr_config.trainer)
     #   asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=asr_config.model, trainer=None)
     #   asr_model.load_state_dict(
     #       torch.load(asr_ckpt_path, map_location=device))
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_conformer_transducer_large")
        asr_model.to(torch.device('cuda:0'))
        asr_model.eval()
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        hypotheses = []
        for spec in tqdm(denoised_specs, desc='Transcribing denoised'):
            spec_t = torch.tensor(spec)
            spec_len = torch.stack([torch.tensor(spec.shape[1]).long()], 0)
            #logits, logits_len, greedy_predictions = asr_model.forward(processed_signal=spec_t[None, :].to(device), processed_signal_length=spec_len.to(device))
            encoded, encoded_len = asr_model.forward(
                processed_signal=spec_t[None, :].to(device), processed_signal_length=spec_len.to(device)
            )
            best_hyp, all_hyp = asr_model.decoding.rnnt_decoder_predictions_tensor(
                encoded,
                encoded_len,
                return_hypotheses=False,
                partial_hypotheses=None,
            )

            hypotheses += best_hyp
    
        hypotheses = normalize_transducer_output(hypotheses)
        df['denoised_prediction'] = hypotheses

        transcripts = asr_model.transcribe(df['input'])[0]
        transcripts = normalize_transducer_output(transcripts)
        df['noisy_prediction'] = transcripts

        df['noisy_prediction'] = df['noisy_prediction'].fillna('')
        df['denoised_prediction'] = df['denoised_prediction'].fillna('')

        pattern = r'[^a-zA-Z\ ]'
        df['text'] = df['text'].apply(lambda x: x.replace("'", " "))
        df['text'] = df['text'].apply(lambda x: re.sub(pattern, '', x.strip().lower()))
        #__import__('ipdb').set_trace()

        df.to_csv(exp_dir + manifest_path.split('/')[-1].split('_txt')[0]  + '_transducer_eval.csv', encoding='utf-8', index=True )

    # exp_manager(trainer, config.get("exp_manager", None))
