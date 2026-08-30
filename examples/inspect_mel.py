"""Look at what the model produces before the vocoder: the encoder's prior μ and the decoded log-mel.

Requires matplotlib (`pip install matplotlib`).
"""

import matplotlib.pyplot as plt
import numpy as np

from vovo_mlx import VovoTTS

tts = VovoTTS.from_pretrained()
text = "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts."
print("phones:", "".join(tts.phonemize(text)))
s = tts.synthesize(text, steps=16, guidance=2.0)
print(f"{len(s.durations)} phones → {s.mel.shape[0]} frames ({s.mel.shape[0] * 256 / 24000:.2f} s)")

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
for ax, (name, mel) in zip(axes, [("prior μ (encoder)", s.prior), ("decoded (flow decoder)", s.mel)]):
    ax.imshow(np.array(mel).T, origin="lower", aspect="auto", vmin=-16.1, vmax=4.0, cmap="viridis")
    ax.set_title(name)
    ax.set_ylabel("mel band")
axes[-1].set_xlabel("frame (256 samples at 24 kHz)")
plt.tight_layout()
plt.savefig("mel.png", dpi=120)
print("wrote mel.png")
