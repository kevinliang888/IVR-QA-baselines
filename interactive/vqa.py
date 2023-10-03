from interactive.extract_feature import get_video_emdeds, get_metric_results, get_inds_sa
import numpy as np
from interactive.gen_cap import *

@torch.no_grad()
def vqa_retrieval(model_vqa, model_retrieval, data_loader, k_test, augment=False, separate=True,
                  num_segment=2, ask_object=False, ask_regular=True, aggregate=True):

    device = torch.device('cuda')
    videos, video_ids, video_embeds, video_feats, video_frame_embeds,\
    video_frame_feat = get_video_emdeds(model_retrieval, data_loader, device=device)
    texts = data_loader.dataset.text

    num_videos = len(videos)
    ranks_new = np.zeros(num_videos)

    for i in range(num_videos):
        video_frame = videos[i].to(device,non_blocking=True)
        origin_caption = texts[i: i + 1][0]
        caption1 = generate_caption_vqa(video_frame, origin_caption, model_vqa)
        all_caps = [origin_caption, caption1]
        if augment:
            generate_caps = generate_augment_caption(video_frame, model_vqa, origin_caption, num_segment=num_segment,
                                                    ask_object=ask_object, ask_regular=ask_regular)
            all_caps.extend(generate_caps)

        all_caps = list(dict.fromkeys(all_caps))

        if separate:
            append_caption = ' [SEP] '.join(all_caps)
        else:
            append_caption = ' and '.join(all_caps)

        aggregate_captions = [origin_caption] if aggregate else []
        inds = get_inds_sa(model_retrieval, video_embeds, video_feats, k_test=k_test,
                           caption=append_caption, aggregate_captions=aggregate_captions)

        ranks_new[i] = np.where(inds == i)[0][0]
        # inds2 = get_inds_sa(model_retrieval, video_embeds, video_feats, k_test=k_test, caption=origin_caption)

    vr1, vr5, vr10, vr_mean, mdR = get_metric_results(ranks_new)
    eval_result = {'vid_r1': vr1,
                   'vid_r5': vr5,
                   'vid_r10': vr10,
                   'vid_r_mean': vr_mean,
                   'mdR': mdR}
    return eval_result


@torch.no_grad()
def vqa_retrieval_auto(model_vqa, model_retrieval, data_loader, k_test, top_k=5, round=1, model_cap=None,
                       config_cap=None, t0_tokenizer=None, t0_model=None, augment=False, use_caption=False):

    device = torch.device('cuda')
    cpu_device = torch.device('cpu')

    videos, video_ids, video_embeds, video_feats, video_frame_embeds,\
    video_frame_feat = get_video_emdeds(model_retrieval, data_loader, device=device)
    texts = data_loader.dataset.text
    num_videos = len(videos)

    ranks_new = np.zeros(num_videos)
    for i in range(num_videos):
        video_frame = videos[i].to(device,non_blocking=True)
        origin_caption = texts[i: i + 1][0]
        append_caption, append_caption2 = origin_caption, origin_caption
        final_captions = [origin_caption]

        for j in range(round):
            inds = get_inds_sa(model_retrieval, video_embeds, video_feats, k_test=k_test, caption=append_caption, aggregate_captions=[])
            top_k_inds = inds[:top_k].copy()
            top_k_video_feats = video_feats[top_k_inds].cuda()

            # Due to GPU memory constraint, put LLM on gpu and others on cpu.
            model_retrieval = model_retrieval.to(cpu_device)
            model_vqa = model_vqa.to(cpu_device)
            t0_model = t0_model.to(device)
            question = generate_question_auto(model_cap, config_cap, top_k_video_feats, append_caption2, t0_tokenizer, t0_model, use_caption=use_caption)

            # Put models back to gpu
            t0_model = t0_model.to(cpu_device)
            model_retrieval = model_retrieval.to(device)
            model_vqa = model_vqa.to(device)

            new_caption = generate_caption_vqa_auto(video_frame, model_vqa, question)
            if augment:
                # Ask Segment
                aug_captions = generate_augment_caption_auto(video_frame, model_vqa, question, num_segment=2)
                aug_captions.insert(0, new_caption)
                aug_captions = set(aug_captions)
                if "" in aug_captions: aug_captions.remove("")
                final_captions.extend(aug_captions)

                # Ask Object
                object_captions = generate_augment_caption(video_frame, model_vqa, origin_caption="",
                                                           num_segment=2, ask_object=True, ask_regular=False)
                final_captions.extend(object_captions)
            else:
                if new_caption != "":
                    final_captions.append(new_caption)

            append_caption = ' [SEP] '.join(final_captions)
            append_caption2 = ', '.join(final_captions)

        inds = get_inds_sa(model_retrieval, video_embeds, video_feats, k_test=k_test,
                           caption=append_caption, aggregate_captions=[])
        ranks_new[i] = np.where(inds == i)[0][0]
        # inds2 = get_inds_sa(model_retrieval, video_embeds, video_feats, k_test=k_test, caption=origin_caption)


    vr1, vr5, vr10, vr_mean, mdR = get_metric_results(ranks_new)
    eval_result = {'vid_r1': vr1,
                   'vid_r5': vr5,
                   'vid_r10': vr10,
                   'vid_r_mean': vr_mean,
                   'mdR': mdR}

    return eval_result
