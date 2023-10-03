import json
from pathlib import Path
import re

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)

def load_msrvtt_dataset(data_path='data/msrvtt_ret/msrvtt_test.json'):
    data = read_json(data_path)
    full_texts = {}
    video_ids = []
    for i in range(len(data)):
        video_name, total_frames, captions = data[i]
        video_id = video_name[:video_name.find('.')]
        full_texts[video_id] = captions
        video_ids.append(video_id)
    index_to_texts = {}
    for i in range(len(video_ids)):
        video_id = video_ids[i]
        index_to_texts[i] = (video_id, full_texts[video_id])
    return index_to_texts

def get_conjunctions(singular_captions, human_caption):
    singular = human_caption in singular_captions
    verb = "is" if singular else "are"
    return verb

def simplify_caption(caption):
    caption = caption.strip('.').lower()
    return caption


def int_to_en(integer):
    d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
         6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}
    return d[integer]