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
import whisper

if __name__ == '__main__':

    
    argparser = ArgumentParser()
    argparser.add_argument('config', type=str, help='A config yaml file, like denoise_example.yml')

    args = argparser.parse_args()

    config = OmegaConf.load(args.config)

    exp_dir = config.expdir_path
    manifest_paths = config.manifest_path
    rec_config_path = config.reconstruction.config_path
    rec_ckpt_path = config.reconstruction.checkpoint_path
   
    device = torch.device('cuda:0')

    for manifest_path in manifest_paths.split(','):

        df = pd.read_json(manifest_path, lines=True)

        #rec_config = OmegaConf.load(rec_config_path)
        #rec_model = nemo_asr.models.ReconstructionWhisperModel(cfg=rec_config.model, trainer=None)
        #chkpt = torch.load(rec_ckpt_path, map_location=device)
        
        #rec_model.load_state_dict(chkpt['state_dict'])
        #rec_model.to(device)
        #denoised_specs = rec_model.reconstruct(df['input'])
        #print(denoised_specs[0].shape)

        #del rec_model

        asr_model = whisper.load_model("base.en")
        asr_model.to(torch.device('cuda:0'))
        options = whisper.DecodingOptions()

        #hypotheses = []
        #for spec in tqdm(denoised_specs, desc='Transcribing denoised'):
        #    spec_t = torch.tensor(spec).to(device)
            #spec_len = torch.stack([torch.tensor(spec.shape[1]).long()], 0)
       #     current_hypotheses = whisper.decode(asr_model, spec_t, options)
       #     hypotheses.append(current_hypotheses.text)
        #__import__('ipdb').set_trace()

        #df['denoised_prediction'] = hypotheses
        
        transcripts = []
        key = ''
        if 'input' in  df.keys():
            key = 'input'
        elif 'audio_filepath' in df.keys():
            key = 'audio_filepath'
        else:
            print(df.keys())
        for file in tqdm(df[key]):
            #audio = whisper.load_audio(file)
            #audio = whisper.pad_or_trim(audio)
            #mel = whisper.log_mel_spectrogram(audio).to(device)
            #transcripts.append(whisper.decode(asr_model, mel, options).text)
            transcript = asr_model.transcribe(file, language="english")["text"]
            pattern = r'[^a-zA-Z\ ]'
            transcript = transcript.replace("'", " ")
            transcript = re.sub(pattern, '', transcript.strip().lower())
            transcripts.append(transcript)
        df['noisy_prediction'] = transcripts
    

        df['noisy_prediction'] = df['noisy_prediction'].fillna('')
       # df['denoised_prediction'] = df['denoised_prediction'].fillna('')

       # pattern = r'[^a-zA-Z\ ]'
       # df['noisy_prediction'] = df['noisy_prediction'].apply(lambda x: x.replace("'", " "))
       # df['noisy_prediuction'] = df['noisy_prediction'].apply(lambda x: re.sub(pattern, '', x.strip().lower()))
        #__import__('ipdb').set_trace()

        df.to_csv(exp_dir + manifest_path.split('/')[-1].split('_txt')[0]  + '_whisper_eval.csv', encoding='utf-8', index=True )

    # exp_manager(trainer, config.get("exp_manager", None))
