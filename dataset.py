from pathlib import Path
import random
import re

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from PIL import Image
import torch

class PatchDataset(Dataset):
    """
    Returns
        image_tensor  - float32, shape (1,160,240), range 0‒1
        meta          - dict with keys
                          'origin'   : '01' or '02'
                          'frame_id' : int, e.g. 34         (t034)
                          'patch_id' : int, e.g. 5          (_005)
                          'path'     : Path object (full path)
    """
    _pat = re.compile(r"t(\d{3})_(\d{3})\.jpg", re.IGNORECASE)

    def __init__(self, roots, transform=None):
        self.records = []            # list of dicts
        self.transform = transform or ToTensor()

        for root in roots:
            root = Path(root)
            origin = root.stem.split('_')[0]    # '01' or '02'
            for jpg in sorted(root.glob("*.jpg")):
                m = self._pat.fullmatch(jpg.name)
                if not m:
                    raise ValueError(f"Unexpected file name: {jpg}")
                frame_id = int(m.group(1))
                patch_id = int(m.group(2))
                self.records.append(
                    {
                        "path": jpg,
                        "origin": origin,
                        "frame_id": frame_id,
                        "patch_id": patch_id,
                    }
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("L")    # grayscale
        img = self.transform(img)                     # ➞ tensor (1,H,W) in 0‒1
        return img, rec.copy()                        # return *copy* of meta


def split_dataset(full_ds, split, SEED):
    """
    split = (train_frac, val_frac, test_frac)
    returns three *new* Dataset objects (subsets of full_ds).
    """
    assert abs(sum(split) - 1.0) < 1e-6
    n = len(full_ds)
    lengths = [int(round(f * n)) for f in split]
    # adjust in case rounding lost / gained an item
    while sum(lengths) > n:
        lengths[-1] -= 1
    while sum(lengths) < n:
        lengths[0] += 1

    generator = torch.Generator().manual_seed(SEED)
    return random_split(full_ds, lengths, generator=generator)


def collate_with_meta(batch):
    """
    Default collate would turn the *dict* meta into a dict of lists (fine),
    but many people prefer to keep meta as a *list of dicts*.
    This collate returns:
        images - float tensor (B,1,160,240)
        meta   - list[dict] length B
    """
    imgs, metas = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(metas)
