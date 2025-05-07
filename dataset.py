import os
import torch
import torchaudio
import whisper
from glob import glob
from tqdm import tqdm

class TIMIT(torch.utils.data.Dataset):
    def __init__(self, scp_file="scp/test.wav.scp", n_mels=80, device='cpu:0'):
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


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, scp_file="scp/test.wav.scp", n_mels=80, device='cpu:0'):
        root = '/disk/scratch/s2522924/LibriSpeech'
        split = 'dev-clean' # temp hard code
        file_list = sorted(glob(os.path.join(root, split, "**/*.flac"), recursive=True))
        trans_list = sorted(glob(os.path.join(root, split, "**/*.trans.txt"), recursive=True))
        self.label_dict = {}
        for trans in trans_list:
            lines = open(trans, 'r').readlines()
            for l in lines:
                fid, text = l.split(' ', 1)
                self.label_dict[fid] = text
        self.alignment_dict = {}
        raw = open(f'ls_alignment_{split}.txt', 'r').readlines()
        for line in raw:
            fname = line.split(' ', 1)[0]
            self.alignment_dict[fname] = eval(line.split(' ', 1)[1])

        self.dataset = []
        for path in file_list:
            fid = path.split('/')[-1].split('.')[0]
            text = self.label_dict[fid]
            ali = self.alignment_dict[fid]
            self.dataset.append((path, text, ali, fid))
        self.n_mels = n_mels
        self.device = device
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        path, text, ali, fid = self.dataset[item]
        audio, sample_rate = torchaudio.load(path)
        audio = audio.squeeze()
        assert sample_rate == 16000
        duration = len(audio.flatten())
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, self.n_mels)
        starts = []
        ends = []
        for i, item in enumerate(ali):
            if item[0] == '' and i == len(ali) - 1:
                continue
            else:
                starts.append(item[1])
                ends.append(item[2])

        return (mel, duration, text, starts, ends, fid)
