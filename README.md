# Whisper Word Aligner
Source code for the paper: [Whisper Has an Internal Word Aligner](https://arxiv.org/abs/2509.09987)

## Setup
```
pip3 install -U openai-whisper
pip3 install librosa
pip3 install num2words
```

## Usage

#### Infer alignments
Example command for aligning with characters on TIMIT. 

It consists of few steps --- (1) tokenizing texts into *characters*, (2) teacher-forcing Whisper-*medium* with the character sequence, 
(3) selecting *top 10* attention maps to extract the final alignments,
(4) evaluating word alignments within a tolerance of *0.05*s (50ms).

```
python infer_ali.py --dataset TIMIT \
              --scp /path/to/scp \
              --model medium \
              --aggr topk \
              --topk 10 \
              --aligned_unit_type char \
              --strict \
              --output_dir results \
              --tolerance 0.05 \
              --medfilt_width 3
```

#### Probe alignments
Example command for probing the oracle heads in Whisper on TIMIT, note that ground truth alignments are needed to pick the oracle heads. 
```
python3 probe_oracle.py --dataset TIMIT \
              --scp /path/to/scp \
              --model medium \
              --aligned_unit_type char \
              --strict \
              --output_dir results \
              --tolerance 0.05 \
              --medfilt_width 3
```

#### Evaluation on saved alignments
If the alignments are saved specifying the `--save_prediction` argument when running `infer_ali.py` (should be a *.pkl file), we can rerun evaluation against the ground truth with different tolerances using the command:
```
python eval_ali.py --pred /path/to/pkl --tolerance 0.05
```

## Data
The utterances to align needs to be processed into a scp file with the format of `<file_id> <path_to_file>` in each line:

```
dr7-mnjm0-sx410 /group/corporapublic/timit/original/test/dr7/mnjm0/sx410.wav
dr7-mnjm0-sx140 /group/corporapublic/timit/original/test/dr7/mnjm0/sx140.wav
dr7-mnjm0-sx230 /group/corporapublic/timit/original/test/dr7/mnjm0/sx230.wav
...
```

## Sample code snippet for running inference on a single utterance
```python
import torch
import torchaudio
from timing import get_attentions, force_align, filter_attention, default_find_alignment
from retokenize import encode, remove_punctuation
import whisper
from whisper.tokenizer import get_tokenizer

AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

DEVICE = 'cuda:0'

# testing sample
sample_audio = "sample/test.wav"

# load model
model = whisper.load_model("medium")
model.to(DEVICE)
options = whisper.DecodingOptions(language="en")
tokenizer = get_tokenizer(model.is_multilingual, language='English')

# process audio to mel 
audio, sample_rate = torchaudio.load(sample_audio)
audio = audio.squeeze()
duration = len(audio.flatten())
audio = whisper.pad_or_trim(audio.flatten())
mel = whisper.log_mel_spectrogram(audio, 80)
mel = mel.to(DEVICE)

# run align
result = whisper.decode(model, mel, options)
transcription = result.text
transcription = remove_punctuation(transcription)
text_tokens = encode(transcription, tokenizer, aligned_unit_type='char') # choose between 'char' or 'subword'
tokens = torch.tensor(
            [
                *tokenizer.sot_sequence,
                tokenizer.no_timestamps,
                *text_tokens,
                tokenizer.eot,
            ]
        ).to(DEVICE)

max_frames = duration // AUDIO_SAMPLES_PER_TOKEN
attn_w, logits = get_attentions(mel, tokens, model, tokenizer, max_frames, medfilt_width=3, qk_scale=1.0)
words, start_times, end_times, ws, scores = force_align(
    attn_w, text_tokens, 
    tokenizer, 
    aligned_unit_type='char', # choose between 'char' or 'subword'
    aggregation='topk', # choose between 'mean' or 'topk'
    topk=10
)

# print word alignment result
for i, word in enumerate(words[:-1]):
  print(f"{start_times[i]:.2f} {end_times[i]:.2f} {word.strip()}")
```
