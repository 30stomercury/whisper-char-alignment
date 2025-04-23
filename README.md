What we have done:
- We are able to select attention maps that look similar to forced alignments with coverage penalty!
- It seems like some attention heads are modelling word-level forced alignments, some are not.
- Extracting character-level attention maps is possible.

Todos:
- [ ] Compared to mean-pooling over all attention maps (baseline), how good is the one pooled over the selected attention maps with coverage penalty?
- [ ] Out of the selected attention maps, can we find the ones closest to word-level alignments?
- [ ] Are there specific heads, consistently modelling word-level alignmetns? -> check more samples.
- [ ] Apply coverage penalty to samples in librspeech dev-clean. Plot word-level alignments.
- [ ] Character-level ground truth alignments with MFA, CTC?
- [ ] Compute character-level inference WERs by simply blocking tokens other than characters.

Advanced:
- [ ] Compute $\sum_{\text{next char}} \arg \max_{x \in A(prefix)} p(\text{next char}| x)$, in which $A$ is the pretrained tokenizer. 


---
Baselines:

WhisperX, averaged-pooling, selected-pooling on
- [ ] TIMIT word-level alignments
- [ ] dev-clean word-level alignments

Character-level alignments
- [ ] Char-based whisper WERs


Ref: \
https://www.isca-archive.org/interspeech_2024/rousso24_interspeech.pdf
