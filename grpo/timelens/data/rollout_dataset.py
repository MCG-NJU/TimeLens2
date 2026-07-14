"""Dataset + collate for off-policy rollout inference (batched left-padded generation)."""

from __future__ import annotations

import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.data.video import build_video_content


def collate_fn(batch, processor):
    messages = [item["messages"] for item in batch]
    annos = [item["anno"] for item in batch]
    texts = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=texts,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        padding=True,
        padding_side="left",
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    return {"inputs": inputs, "annos": annos}


class RolloutDataset(Dataset):
    def __init__(self, annos, data_args):
        super().__init__()
        self.annos = annos
        self.data_args = data_args

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])
        message = {
            "role": "user",
            "content": [
                build_video_content(anno, self.data_args, include_video_range=False),
                {"type": "text", "text": anno["query"]},
            ],
        }
        return {"messages": [message], "anno": anno}
