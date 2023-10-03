import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.blip_retrieval import blip_retrieval
from models.blip import blip_decoder
from models.blip_vqa import blip_vqa
import utils
from interactive.vqa import vqa_retrieval, vqa_retrieval_auto
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from data.dataloader_didemo_retrieval import dataloader_didemo_test


def main(args, config, config_vqa, config_cap=None):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating retrieval dataset")

    #### Model ####
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'])

    model = model.to(device)
    test_loader = dataloader_didemo_test(args, model.tokenizer, subset="test")
    model_vqa = blip_vqa(pretrained=config_vqa['pretrained'], image_size=config_vqa['image_size'],
                         vit=config_vqa['vit'], vit_grad_ckpt=config_vqa['vit_grad_ckpt'],
                         vit_ckpt_layer=config_vqa['vit_ckpt_layer'])
    model_vqa = model_vqa.to(device)

    model_cap, t0_tokenizer, t0_model = None, None, None
    if args.automatic or args.half_auto:
        model_cap = blip_decoder(pretrained=config_cap['pretrained'], image_size=config_cap['image_size'],
                                 vit=config_cap['vit'], prompt=config_cap['prompt'])
        model_cap = model_cap.to(device)
        t0_tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp", cache_dir="./t0_model")
        t0_model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", cache_dir="./t0_model")
        # t0_model = t0_model.to(device)

    if args.automatic:
        eval_results = vqa_retrieval_auto(model_vqa, model, test_loader, config['k_test'], top_k=args.top_k, round=args.round,
                                          model_cap=model_cap, config_cap=config_cap, t0_tokenizer=t0_tokenizer,
                                          t0_model=t0_model, augment=args.augment, use_caption=args.use_caption)
    else:
        eval_results = vqa_retrieval(model_vqa, model, test_loader, config['k_test'],
                                     separate=args.separate, fix_grammar=args.fix_grammar, augment=args.augment,
                                     num_segment=args.num_segment, ask_object=args.ask_object, top_k=args.top_k,
                                     ask_regular=args.ask_regular, aggregate=args.aggregate, automatic=args.half_auto,
                                     model_cap=model_cap, config_cap=config_cap, t0_tokenizer=t0_tokenizer, t0_model=t0_model)
    print(eval_results)
    log_stats = {**{f'{k}': v for k, v in eval_results.items()}, }

    with open(os.path.join(args.output_dir, f"test_results.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrieval_config', default='./configs/retrieval_msrvtt.yaml')
    parser.add_argument('--vqa_config', default='./configs/vqa.yaml')
    parser.add_argument('--cap_config', default='./configs/nocaps.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_msrvtt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--fix_grammar', action='store_true', default=False)
    parser.add_argument('--separate', action='store_true', default=False)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--num_segment', default=2, type=int)
    parser.add_argument('--ask_object', action='store_true', default=False)
    parser.add_argument('--ask_regular', action='store_true', default=False)
    parser.add_argument('--aggregate', action='store_true', default=False)
    parser.add_argument('--automatic', action='store_true', default=False)
    parser.add_argument('--half_auto', action='store_true', default=False)
    parser.add_argument('--use_caption', action='store_true', default=False)
    parser.add_argument('--round', default=1, type=int)
    parser.add_argument('--top_k', default=4, type=int)
    parser.add_argument('--partial', action='store_true', default=False)
    parser.add_argument('--data_path', default="/n/fs/kl-project/BLIP/data/didemo_ret")
    parser.add_argument('--features_path', default="/n/fs/visualai-scr/kaiqu/didemo/videos/test_videos")

    args = parser.parse_args()
    config = yaml.load(open(args.retrieval_config, 'r'), Loader=yaml.Loader)
    config_vqa = yaml.load(open(args.vqa_config, 'r'), Loader=yaml.Loader)
    config_cap = yaml.load(open(args.cap_config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    main(args, config, config_vqa, config_cap)