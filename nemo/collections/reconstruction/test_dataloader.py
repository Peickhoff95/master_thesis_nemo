from nemo.collections.speech_features.data.audio_to_audio import AudioToAudioDataset

if __name__ == "__main__":
    manifest_files = [
        './toydata/nemo/train.json',
        './toydata/nemo/test.json',
        './toydata/nemo/dev.json'
    ]
    manifest_files = ','.join(manifest_files)
    collection = AudioToAudioDataset(manifest_files, 16000)
    __import__('ipdb').set_trace()
