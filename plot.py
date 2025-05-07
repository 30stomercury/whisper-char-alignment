import matplotlib.pyplot as plt

def plot_attns(attns, scores, wrd_pos, path='imgs'):
    wrd_pos = [int(i/0.02) for i in wrd_pos]
    if scores is not None:
        names = []
        for score, (l, n_h), name in scores:
            names.append(name)
        for attn, name in zip(attns, names):
            plt.vlines(wrd_pos, ymin=-1, ymax=attn.size(0), colors="white")
            plt.imshow(attn, aspect="auto")
            plt.savefig(f"{path}/{name}.png")
    ws_ = torch.cat(attns, 0).mean(0)
    plt.vlines(wrd_pos, ymin=-1, ymax=ws_.size(0), colors="white")
    plt.imshow(ws_, aspect="auto")
    plt.savefig(f"{path}/sample_ave.png")

