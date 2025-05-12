import os
import torch
import torchaudio
import whisper
from glob import glob
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from datasets import load_dataset, Audio
import xml.etree.ElementTree as ET




class Collate: 
    def __call__(self, batch):
        one_batch = list(zip(*batch))
        audio, mel, duration, text, starts, ends, fid = one_batch
        return  audio[0], mel[0], duration[0], text[0], starts[0], ends[0], fid[0]


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

        return audio, mel, duration, text, starts, ends, fid

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


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, scp_file="scp/dev-clean.wav.scp", n_mels=80, device='cpu:0'):
        # root = '/disk/scratch/s2522924/LibriSpeech'
        # split = 'dev-clean' # temp hard code
        # file_list = sorted(glob(os.path.join(root, split, "**/*.flac"), recursive=True))
        scp = open(scp_file, 'r').readlines()
        split = scp[0].split(' ')[1].split('/')[-4]
        root = scp[0].split(' ')[1].split(split)[0]
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
        for line in scp:
            splits = line.split()
            fid, path = splits[0], splits[1]
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

        return (audio, mel, duration, text, starts, ends, fid)




class AMI(torch.utils.data.Dataset):
    def __init__(self, data_dir="/home/s2196654/dataset/AMI/", n_mels=80, device='cpu:0'):
        subset = 'ihm'
        dataset = load_dataset('edinburghcstr/ami', subset, split='test')

        spk_mapping = self.load_speaker_mapping(data_dir)
        all_meetings = {}
        for meeting_id in set(dataset['meeting_id']):
            for k, v in spk_mapping[meeting_id].items():
                all_meetings[f"{meeting_id}.{v}"] = self.load_meeting(data_dir, meeting_id, v)


        meeting_clips = defaultdict(list)
        for datapoint in dataset:
            meeting_id = datapoint['meeting_id']
            spk_id = datapoint['speaker_id']
            # spk in A-D
            spk = spk_mapping[meeting_id][spk_id]
            meeting_spk_id = f"{datapoint['meeting_id']}.{spk}"
            if meeting_spk_id not in ['EN2002a.A']: #, 'EN2002a.B', 'EN2002a.C', 'EN2002a.D']:
                continue
            meeting_clips[meeting_spk_id].append(
                (
                    datapoint['begin_time'], 
                    datapoint['end_time'], 
                    datapoint['text'], 
                    datapoint['audio']['path']
                )
            )

        output = self.get_clip_alignments(meeting_clips, all_meetings)
        print("total clips:", sum([len([i[-1] for i in output[x]]) for x in output]))

        self.sample_rate = 16000
        self.dataset = []
        clip_id = 0
        
        for meeting_spk_id in output:
            for (path, s, e, wrds, clip_to_wrd_alignments) in output[meeting_spk_id]:
                if len(clip_to_wrd_alignments) == 0:
                    continue
                starts = []
                ends = []
                text = []
                for (wrd_s, wrd_e, wrd_gt) in clip_to_wrd_alignments:
                    # make it to relative timestamps by substracting s
                    starts.append(wrd_s-s)
                    ends.append(wrd_e-s)
                    text.append(wrd_gt)
                fid = meeting_spk_id + f'.{clip_id}'
                clip_id += 1
                self.dataset.append((path, self.sample_rate, text, starts, ends, fid))
        self.n_mels = n_mels
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio_file, sample_rate, text, starts, ends, fid = self.dataset[item]
        audio, sample_rate = torchaudio.load(audio_file)
        assert sample_rate == self.sample_rate
        duration = len(audio.flatten())
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio, self.n_mels)
        mel = mel.to(self.device)

        return audio, mel, duration, text, starts, ends, fid

    def load_speaker_mapping(self, data_dir):
        target = f"{data_dir}/corpusResources/meetings.xml"
        x = ET.parse(target)
        root = x.getroot()

        spk_mapping = {}
        # Iterate over all 'speaker' elements inside 'meeting'
        for meeting in root.findall("meeting"):
            meeting_id = meeting.attrib.get("observation")
            spk_mapping[meeting_id] = {}
            for speaker in meeting.findall("speaker"):
                global_name = speaker.attrib.get("global_name")
                nxt_agent = speaker.attrib.get("nxt_agent")
                if global_name and nxt_agent:
                   spk_mapping[meeting_id][global_name] = nxt_agent

        return spk_mapping

    def load_meeting(self, data_dir, meeting_id, spk):
        # e.g., '/path//AMI/words/EN2001a.B.words.xml'
        # --> '/home/s2196654/dataset/AMI/words/EN2001a.B.words.xml'
        target_meeting = f"{data_dir}/words/{meeting_id}.{spk}.words.xml"
        x = ET.parse(target_meeting)
        root = x.getroot()
        namespace = {'nite': 'http://nite.sourceforge.net/'}

        # Extract the data
        word_tuples = []
        for w in root.findall('w', namespaces=namespace) + root.findall('nite:w', namespaces=namespace):
            punc = False
            if 'punc' in w.attrib:
                punc = bool(w.attrib['punc'])
            start = float(w.attrib['starttime'])
            end = float(w.attrib['endtime'])
            text = w.text.strip() if w.text else ""
            # Skip punctuations
            if text and not punc:  
                word_tuples.append((start, end, text))
        return word_tuples


    def find_start_point(self, s, wrd_alignments, wrd_id):
        wrd_gt_s = wrd_alignments[wrd_id][0]
        while abs((s - wrd_gt_s)) > 0.005:
            wrd_id += 1
            if len(wrd_alignments) == wrd_id:
                return -1
            wrd_gt_s = wrd_alignments[wrd_id][0]
        return wrd_id

    def get_clip_alignments(self, meeting_clips, all_wrd_alignments):
        # convert from clip alignments to wrd-level alignments
        output = defaultdict(list)
        for meeting_spk_id in meeting_clips:
            meeting_clips[meeting_spk_id].sort()
            clip_alignments = meeting_clips[meeting_spk_id]
            wrd_alignments = all_wrd_alignments[meeting_spk_id]

            # find where to start
            wrd_id = 0

            # align
            for (s, e, wrds, audio) in clip_alignments:
                clip_to_wrd_alignments = []

                wrd_id_tmp = wrd_id
                wrd_id = self.find_start_point(s, wrd_alignments, wrd_id)
                if wrd_id == -1:
                    wrd_id = wrd_id_tmp
                    print("No alignments")
                    continue

                if len(wrds.split()) == 1:
                    continue
                for w in wrds.split():
                    wrd_s, wrd_e, wrd_gt = wrd_alignments[wrd_id]
                    wrd_gt = wrd_gt.upper()
                    if w != wrd_gt:
                        print(w, wrd_gt)
                        clip_to_wrd_alignments = []
                        break
                    wrd_id += 1
                    clip_to_wrd_alignments.append((wrd_s, wrd_e, wrd_gt))
                    
                output[meeting_spk_id].append((audio, s, e, wrds, clip_to_wrd_alignments))
        return output

