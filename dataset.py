import torch
import torchaudio
import whisper

class TIMIT(torch.utils.data.Dataset):
    def __init__(self, scp_file="scp/test.wav.scp", split="test", n_mels=80, device='cpu:0'):
        self.sample_rate = 16000
        self.dataset = []
        scp = open(scp_file, 'r').readlines()
        for line in scp:
            splits = line.split()
            fid = splits[0]
            # audio
            audio_file = splits[1]
            audio, sample_rate = torchaudio.load(audio_file)
            audio = audio.squeeze()
            # text
            text_file = audio_file.split('.wav')[0] + '.wrd'
            text, starts, ends = self.process_text(text_file)
            self.dataset.append((audio, sample_rate, text, starts, ends, fid))
        self.n_mels = n_mels
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, starts, ends, fid = self.dataset[item]
        assert sample_rate == self.sample_rate
        duration = len(audio.flatten())
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, self.n_mels)
        mel = mel.to(self.device)

        return mel, duration, text, starts, ends, fid

    def process_text(self, filename):
        starts = []
        ends = []
        texts = []
        f = open(filename, 'r')
        for line in f.readlines():
            splits = line.split()
            starts.append(float(splits[0])/self.sample_rate)
            ends.append(float(splits[1])/self.sample_rate)
            texts.append(splits[2])
        texts = " ".join(texts)
        return texts, starts, ends

class Collate: 
    def __call__(self, batch):
        one_batch = list(zip(*batch))
        mel, duration, text, starts, ends, fid = one_batch
        return  mel[0], duration[0], text[0], starts[0], ends[0], fid[0]
