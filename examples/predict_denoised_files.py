import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from tqdm.auto import tqdm

if __name__ == '__main__':

    df = pd.read_json('/home/patrick/Projects/Datasets/NSD/trainset_56spk_txt.json', lines=True)

    rec_config = OmegaConf.load(
        '/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small_homepc.yml')
    rec_model = nemo_asr.models.ReconstructionModel(cfg=rec_config.model, trainer=None)
    chkpt = torch.load(
        '/home/patrick/Projects/master_thesis_nemo/experiments/2022-07-09_03-09-18/checkpoints/Conformer-Reconstruction--val_loss=197058.5781-epoch=90.ckpt')

    rec_model.load_state_dict(chkpt['state_dict'])
    denoised_specs = rec_model.reconstruct(df['input'])

    del rec_model

    asr_config = OmegaConf.load(
        '/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_asr_homepc.yml')
    asr_config.trainer.progress_bar_refresh_rate = 10

    # trainer = pl.Trainer(**asr_config.trainer)
    asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=asr_config.model, trainer=None)
    asr_model.load_state_dict(
        torch.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small.pt'))
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
    __import__('ipdb').set_trace()

    df.to_csv('/home/patrick/Projects/master_thesis_nemo/experiments/2022-07-09_03-09-18/trainset_56spk_eval.csv', encoding='utf-8', index=True )

    # exp_manager(trainer, config.get("exp_manager", None))
