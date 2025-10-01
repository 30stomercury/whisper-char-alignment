# Whisper Word Aligner
Source code for the paper: [Whisper Has an Internal Word Aligner](https://arxiv.org/abs/2509.09987)

## Setup
```
pip3 install -U openai-whisper
pip3 install datasets
pip3 install librosa
```

## Usage
Example command for aligning with characters on TIMIT
```
python test.py --dataset TIMIT \
              --scp /path/to/scp \
              --aggr topk \
              --topk 10 \
              --aligned_unit_type char \
              --strict \
              --output_dir results \
              --tolerance 0.05 \
              --medfilt_width 3
```
