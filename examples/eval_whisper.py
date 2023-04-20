import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    asr_config = OmegaConf.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_medium_asr_homepc.yml')
    asr_config.trainer.progress_bar_refresh_rate = 10

    #trainer = pl.Trainer(**asr_config.trainer)
    asr_model = nemo_asr.models.EncDecCTCModelBPE(cfg=asr_config.model, trainer=None)
    asr_model.load_state_dict(torch.load('/home/patrick/Projects/master_thesis_nemo/pretrained_models/stt_en_conformer_ctc_medium.pt'))

    rec_config = OmegaConf.load(
        '/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_medium_asr_homepc.yml')
    rec_model = nemo_asr.models.ReconstructionModel(cfg=rec_config.model, trainer=None)
    chkpt = torch.load('/home/patrick/Projects/master_thesis_nemo/experiments/Conformer-Reconstruction-Medium/2022-07-08_02-14-11/checkpoints/Conformer-Reconstruction-Medium--val_loss=194403.6875-epoch=98.ckpt')

    rec_model.load_state_dict(chkpt['state_dict'])
    #exp_manager(trainer, config.get("exp_manager", None))

    asr_model.cuda()
    asr_model.eval()
    rec_model.cuda()
    rec_model.eval()
    wer_nums = []
    wer_denoms = []

    for test_batch in rec_model.test_dataloader():
        test_batch = [x.cuda() for x in test_batch]
        targets = test_batch[3]
        targets_lengths = test_batch[4]
        processed_signal, processed_signal_len = rec_model(input_signal=test_batch[0],input_signal_length=test_batch[2])
        #__import__('ipdb').set_trace()
        log_probs, encoded_len, greedy_predictions = asr_model(
            processed_signal=processed_signal, processed_signal_length=processed_signal_len
        )
        #log_probs, encoded_len, greedy_predictions = asr_model(
        #    processed_signal=test_batch[0], processed_signal_length=test_batch[2]
        # )
        # Notice the model has a helper object to compute WER
        asr_model._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = asr_model._wer.compute()
        asr_model._wer.reset()
        wer_nums.append(wer_num.detach().cpu().numpy())
        wer_denoms.append(wer_denom.detach().cpu().numpy())

        # Release tensors from GPU memory
        del test_batch, processed_signal_len, processed_signal, log_probs, targets, targets_lengths, encoded_len, greedy_predictions

    # We need to sum all numerators and denominators first. Then divide.
    print(f"WER = {sum(wer_nums) / sum(wer_denoms)}")

