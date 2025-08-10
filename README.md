#### Whisperacter word aligner

Alignments:
- [LibriSpeech alignments](https://zenodo.org/records/2619474#.YnB_1fPMK3I) (TextGrid)
- [processed LibriSpeech alignments](https://drive.google.com/drive/folders/10Qa8dedfFhVl-3NuxMQMUUOwo9Rwn33o?usp=sharing) (merge TextGrid to single txt file)

```
pip3 install datasets
pip3 install librosa
export HF_HUB_OFFLINE=1
python3 test.py --dataset AMI --output_dir results_debug/ --scp ../dataset/AMI/
```
