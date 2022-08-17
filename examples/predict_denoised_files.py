import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
import re
import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':

    exp_dir = '/home/patrick/Projects/master_thesis_nemo/experiments/Conformer-Reconstruction-Unfrozen/2022-08-15_15-08-16/'
    manifest_path = '/home/patrick/Projects/Datasets/NSD/testset_txt.json'

    df = pd.read_json(manifest_path, lines=True)

    rec_config = OmegaConf.load(
        '/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small_homepc.yml')
    rec_model = nemo_asr.models.ReconstructionModel(cfg=rec_config.model, trainer=None)
    chkpt = torch.load(
        exp_dir + 'checkpoints/Conformer-Reconstruction-Unfrozen--val_loss=92888.3594-epoch=238.ckpt')

    rec_model.load_state_dict(chkpt['state_dict'])
    denoised_specs = rec_model.reconstruct(df['input'])

    del rec_model

    asr_config = OmegaConf.load(
        '/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_medium_asr_homepc.yml')
    asr_config.trainer.progress_bar_refresh_rate = 10

    # trainer = pl.Trainer(**asr_config.trainer)
    asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=asr_config.model, trainer=None)
    asr_model.load_state_dict(
        torch.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/stt_en_conformer_ctc_medium.pt'))
    asr_model.eval()
    asr_model.encoder.freeze()
    asr_model.decoder.freeze()

    hypotheses = []
    for spec in tqdm(denoised_specs, desc='Transcribing denoised'):
        spec_t = torch.tensor(spec)
        spec_len = torch.stack([torch.tensor(spec.shape[1]).long()], 0)
        logits, logits_len, greedy_predictions = asr_model.forward(processed_signal=spec_t[None, :], processed_signal_length=spec_len)
        current_hypotheses = asr_model._wer.ctc_decoder_predictions_tensor(
            greedy_predictions, predictions_len=logits_len, return_hypotheses=False,
        )
        hypotheses += current_hypotheses

    df['denoised_prediction'] = hypotheses

    transcripts = asr_model.transcribe(df['input'])
    df['noisy_prediction'] = transcripts

    df['noisy_prediction'] = df['noisy_prediction'].fillna('')
    df['denoised_prediction'] = df['denoised_prediction'].fillna('')

    pattern = r'[^a-zA-Z\ ]'
    df['text'] = df['text'].apply(lambda x: x.replace("'", " "))
    df['text'] = df['text'].apply(lambda x: re.sub(pattern, '', x.strip().lower()))
    #__import__('ipdb').set_trace()

    df.to_csv(exp_dir + manifest_path.split('/')[-1].split('_txt')[0]  + '_eval.csv', encoding='utf-8', index=True )

    # exp_manager(trainer, config.get("exp_manager", None))
