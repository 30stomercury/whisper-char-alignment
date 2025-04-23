import torch
import whisper
from whisper.tokenizer import get_tokenizer
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

def coverage_penalty(attn, threshold=0.5):
    """
    attn : torch.tensor in (tokens, frames)
    """

    coverage = torch.sum(attn, dim=0)

    # Compute coverage penalty
    penalty = torch.max(
        coverage, coverage.clone().fill_(threshold)
    ).sum(-1)
    penalty = penalty - coverage.size(-1) * threshold
    return penalty

def count_transitions(x):
    count = 0
    positions = []
    prev = x[0]
    for i in range(1, len(x)):
        if x[i] != x[i-1]: 
            positions.append(i)
            count += 1

    return count, positions


AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

# basically paremeters to do denoising
medfilt_width = 7
qk_scale = 1.0

model = whisper.load_model("medium")
tokenizer = get_tokenizer(model.is_multilingual, language='English')


# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("/home/s2196654/dataset/timit/timit-wav/train/dr1/fcjf0/sa1.wav")
duration = len(audio)
audio = whisper.pad_or_trim(audio)
transcription = "She had your dark suit in greasy wash water all year"

# audio
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# tokens

#transcription = ' '.join([i for i in transcription])
tokens = torch.tensor(
    [
        *tokenizer.sot_sequence,
        tokenizer.timestamp_begin,
    ] + tokenizer.encode(transcription) + [
        tokenizer.timestamp_begin + duration // AUDIO_SAMPLES_PER_TOKEN,
        tokenizer.eot,
    ]
).cuda()
phns = "sil sil sil sil sil sil sil sil sil sil sil sil sil sil sil sil sil sil sil sh sh sh sh sh sh sh sh sh ih ih ih ih ih ih ih hh hh hh hh hh hh eh eh eh eh eh eh eh eh eh eh eh eh eh sil sil sil jh jh jh jh jh jh jh ih ih ih ih ih ih ih sil sil sil sil sil sil sil d ah ah ah ah ah ah ah ah ah ah ah ah sil sil sil sil sil sil sil sil k k k s s s s s s s s s s s uw uw uw uw uw uw uw uw uw uw uw uw uw uw unk unk unk unk unk n n n n n n n n n sil sil g g r r r r r r ih ih ih ih ih ih ih ih s s s s s s s s s s ih ih ih ih ih ih w w w w w w w w w w aa aa aa aa aa aa aa aa aa aa aa aa aa sh sh sh sh sh sh sh sh sh sh sil sil w w w w w w aa aa aa aa aa aa aa aa aa dx dx er er er er er er er aa aa aa aa aa aa aa aa aa aa aa aa aa l l l l y y y y y y y y y y y ih ih ih ih ih ih ih ih ih er er er er er er er sil sil sil sil sil sil sil sil sil sil sil sil sil sil".split()[::2]

_, pos = count_transitions(phns)
wrd_pos = [10, 18, 27, 29, 36, 39, 51, 64, 71, 72, 88, 104, 105, 117, 126, 139, 139]


# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")


# install hooks on the cross attention layers to retrieve the attention weights
# NOTE: make sure MultiHeadAttention.use_sdpa = False
QKs = [None] * model.dims.n_text_layer

for i, block in enumerate(model.decoder.blocks):
    block.cross_attn.register_forward_hook(
        lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
    )

with torch.no_grad():
    logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))


weights = torch.cat(QKs)  # layers * heads * tokens * frames    
weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()
weights = median_filter(weights, (1, 1, 1, medfilt_width))
weights = torch.tensor(weights * qk_scale).softmax(dim=-1)
w = weights / weights.norm(dim=-2, keepdim=True)

# whisper implementation:
matrix = w[-6:].mean(axis=(0, 1))
plt.imshow(matrix, aspect="auto")
plt.savefig(f"imgs/sample_ave.png")


scores = []
for l in range(w.size(0)):
    for n_h in range(w.size(1)):
        score = coverage_penalty(w[l, n_h])
        name = f"sample_layer{l}_head{n_h}"
        print(name, score)
        scores.append((score, (l, n_h), name))

topk = 20
scores_sorted = sorted(scores)[:topk]
import pdb; pdb.set_trace()
ws = []
for _, (l, n_h), name in scores_sorted:
    plt.vlines(wrd_pos, ymin=-1, ymax=w[l, n_h].size(0), colors="white")
    plt.imshow(w[l, n_h], aspect="auto")
    # y_max = # tokens
    name = f"sample_layer{l}_head{n_h}"
    plt.savefig(f"imgs_char_topk/{name}.png")
    ws.append(w[l, n_h].unsqueeze(0))

ws = torch.cat(ws, 0).mean(0)
plt.vlines(wrd_pos, ymin=-1, ymax=w[l, n_h].size(0), colors="white")
plt.imshow(ws, aspect="auto")
plt.savefig(f"imgs_char_topk/sample_ave.png")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
