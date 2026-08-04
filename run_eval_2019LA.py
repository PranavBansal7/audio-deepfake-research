"""
Evaluate the pretrained wav2vec2-XLS-R + AASIST checkpoint on the
ASVspoof2019 LA EVAL partition (in-domain), computing EER.

main_SSL_LA.py's --eval path is hardcoded for the ASVspoof2021 LA eval set
(different protocol filename, different audio folder), so this script
reuses the repo's own Model / dataset / EER code against 2019 LA eval instead.
"""
import os
import argparse
import numpy as np
import torch

from model import Model
from data_utils_SSL import genSpoof_list, Dataset_ASVspoof2021_eval as EvalDataset
from eval_metric_LA import compute_eer

# ---- EDIT IF YOUR LAYOUT DIFFERS ----
DATABASE_PATH   = "database/LA/ASVspoof2019_LA_eval/"   # must contain a flac/ subfolder
PROTOCOL_PATH   = "database/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
CHECKPOINT_PATH = "pretrained_models/best_SSL_model_LA.pth"
SCORES_OUT      = "eval_CM_scores_2019_LA_eval.txt"
LIMIT           = None   # set to e.g. 50 for a ~30s smoke test, then back to None for the full run
# --------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# genSpoof_list's is_eval=True branch assumes one bare utt-id per line (the
# 2021-style trial list). The 2019 LA eval protocol ships in the standard
# 5-column labeled format, so we use is_train=False/is_eval=False instead --
# it parses that format AND hands back the ground-truth labels for EER.
d_label_eval, file_eval = genSpoof_list(dir_meta=PROTOCOL_PATH, is_train=False, is_eval=False)
if LIMIT:
    file_eval = file_eval[:LIMIT]
print("Eval utterances:", len(file_eval))

eval_set = EvalDataset(list_IDs=file_eval, base_dir=DATABASE_PATH)

args = argparse.Namespace()  # Model() takes an args param but never reads it
model = Model(args, device).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print("Loaded checkpoint:", CHECKPOINT_PATH)

loader = torch.utils.data.DataLoader(eval_set, batch_size=10, shuffle=False, drop_last=False)
scores_by_id = {}
with torch.no_grad():
    for batch_x, utt_ids in loader:
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()  # class 1 = bonafide
        for uid, s in zip(utt_ids, batch_scores.tolist()):
            scores_by_id[uid] = s

with open(SCORES_OUT, "w") as fh:
    for uid, s in scores_by_id.items():
        fh.write(f"{uid} {s}\n")
print("Scores saved to", SCORES_OUT)

bona = np.array([scores_by_id[u] for u in file_eval if d_label_eval[u] == 1])
spoof = np.array([scores_by_id[u] for u in file_eval if d_label_eval[u] == 0])
print(f"bonafide: {len(bona)}  spoof: {len(spoof)}")

eer, threshold = compute_eer(bona, spoof)
print(f"\nEER on ASVspoof2019 LA eval: {eer*100:.2f}%  (decision threshold={threshold:.4f})")
if LIMIT:
    print("(Smoke test on a subset -- not a real EER. Set LIMIT=None and rerun for the full pass.)")
else:
    print("Verified target for this checkpoint on this exact partition: ~0.2%")
