import ipdb
import torch
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.models import ReconstructionModel
from nemo.collections.asr.modules import ReconstructionEncoder
from omegaconf import OmegaConf

from nemo.collections.speech_features.data.audio_to_audio import AudioToAudioDataset

def main():

    #model = EncDecCTCModel.restore_from(restore_path="~/informatik2/projects/NeMo/pretrained_models/stt_en_conformer_ctc_small.nemo")
    #ipdb.set_trace()
    #model.transcribe(
    #    ["~/informatik2/datasets/LibriSpeech/dev-clean/1272/135031/1272-135031-0001.flac"],
    #    batch_size=1, logprobs=True)
    #exit()
    #model = ReconstructionModel.restore_from(restore_path="/informatik2/students/home/4eickhof/projects/NeMo/pretrained_models/stt_en_conformer_ctc_small.nemo",
    #                                    #override_config_path="/informatik2/students/home/4eickhof/projects/NeMo/examples/reconstruction/conf/conformer_ctc_reconstruction.yaml",
    #                                    strict=False,
    #                                    return_config=False,)
    #ipdb.set_trace()
    #manifest_path = "/export/scratch/4eickhof/datasets/testset_txt.json"
    #ds = AudioToAudioDataset(
    #    manifest_path,
    #    16000,
    #)

    conf = OmegaConf.load("/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small_homepc.yml")

    model = ReconstructionModel(cfg=conf.model)
    model.load_state_dict(torch.load("/home/patrick/Projects/master_thesis_nemo/pretrained_models/conformer_ctc_small.pt",map_location=torch.device('cpu')),
                          strict=False,
    )
    tdl = model._train_dl
    batch = next(iter(tdl))
    tds = tdl.dataset
    item = tds.__getitem__(0)
    ipdb.set_trace()
    prediction = model.forward(input_signal=item[0][None,:],input_signal_length=item[1][None])
    model.loss(x_mag=prediction[0].transpose(1,2),y_mag=item[1][None,:].transpose(1,2),input_lengths=item[1][None])

    print(item)
    #Save state_dict and config
    #torch.save(model.state_dict(), "/informatik2/students/home/4eickhof/conformer_ctc_small.pt")
    #with open("/informatik2/students/home/4eickhof/conformer_ctc_small.yml", "w") as outfile:
    #    OmegaConf.save(config=model, f=outfile)
    #print(model.transcribe(["/informatik2/students/home/4eickhof/datasets/LibriSpeech/dev-clean/1272/135031/1272-135031-0001.flac"], batch_size=1, logprobs=True))


if __name__ == '__main__':
    main()
