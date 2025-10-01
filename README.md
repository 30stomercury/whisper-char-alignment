### Whisperacter word aligner

Alignments:
- [LibriSpeech alignments](https://zenodo.org/records/2619474#.YnB_1fPMK3I) (TextGrid)
- [processed LibriSpeech alignments](https://drive.google.com/drive/folders/10Qa8dedfFhVl-3NuxMQMUUOwo9Rwn33o?usp=sharing) (merge TextGrid to single txt file)

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
