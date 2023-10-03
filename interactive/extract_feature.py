import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import utils
import pdb
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min


def get_disc_feats(top_k_video_embeds, top_k_inds, video_feats, n_cluster=3):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(top_k_video_embeds)
    closest_1, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, top_k_video_embeds)
    index = top_k_inds[closest_1]
    return video_feats[index].cuda()

@torch.no_grad()
def get_video_emdeds(model, data_loader, device='cuda'):
    video_feats = []
    video_embeds = []
    video_frame_embeds = []
    videos = []
    video_ids = []
    for video, video_id in tqdm(data_loader):
        # video shape: [BATCH_SIZE=16, NUM_FRAME=8, CHANNEL, WIDTH, HEIGHT]
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H)
        # video shape: [128, 3, 384, 384]
        video = video.to(device,non_blocking=True)
        video_feat = model.visual_encoder(video)
        video_frame_feat = video_feat
        # video_feat shape: [128, 577, 768]
        video_embed = model.vision_proj(video_feat[:,0,:])
        video_frame_embeds.append(video_embed.view(B*N,-1))
        video_embed = video_embed.view(B,N,-1).mean(dim=1)
        video_embed = F.normalize(video_embed,dim=-1)
        # video_embed shape: [16, 256]
        video_feat = video_feat.view(B,-1,video_feat.shape[-1])
        # video_feat shape: [16, 4616, 768]
        video_feats.append(video_feat.cpu())
        video_embeds.append(video_embed)
        videos.append(video.view((B, N, C, W, H)).cpu())
        video_ids.extend(video_id)
    video_feats = torch.cat(video_feats,dim=0)
    video_embeds = torch.cat(video_embeds,dim=0)
    videos = torch.cat(videos, dim=0)
    video_frame_embeds = torch.cat(video_frame_embeds, dim=0)
    return videos, video_ids, video_embeds, video_feats, video_frame_embeds, video_frame_feat

def get_single_video_embed(model, video, device='cuda'):
    N, C, W, H = video.size()
    video = video.to(device, non_blocking=True)
    video_feat = model.visual_encoder(video)
    # video_feat shape: [8, 577, 768]
    video_embed = model.vision_proj(video_feat[:, 0, :])
    video_embed = video_embed.view(N, -1).mean(dim=0)
    video_embed = F.normalize(video_embed, dim=-1)
    # video_embed shape: [256]
    video_feat = video_feat.view(-1, video_feat.shape[-1])
    # video_feat shape: [4616, 768]
    return video_embed, video_feat

def get_single_text_embed(model, caption, device='cuda'):
    tokenizer = model.tokenizer
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
    return text_embed

def get_text_embeds(model, texts, text_bs=256, device='cuda'):
    tokenizer = model.tokenizer
    num_text = len(texts)
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = tokenizer.additional_special_tokens_ids[0]
    return text_embeds, text_ids, text_atts

def get_v2t_score_matrix(sims_matrix, video_feats, text_ids, text_atts, model, k_test, device='cuda'):
    score_matrix_v2t = torch.full((len(sims_matrix), sims_matrix.shape[1]), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)
    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        encoder_output = video_feats[start + i].repeat(k_test, 1, 1).to(device, non_blocking=True)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device, non_blocking=True)
        output = model.text_encoder(text_ids[topk_idx],
                                    attention_mask=text_atts[topk_idx],
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_v2t[start + i, topk_idx] = score + topk_sim
    return score_matrix_v2t

def get_t2v_score_matrix(sims_matrix, video_feats, text_ids, text_atts, model, k_test, device='cuda'):
    score_matrix_t2v = torch.full((len(sims_matrix),sims_matrix.shape[1]),-100.0).to(device)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()

    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)
    for i,sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        topk_idx = topk_idx.cpu()
        encoder_output = video_feats[topk_idx].to(device,non_blocking=True)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True)
        output = model.text_encoder(text_ids[start+i].repeat(k_test,1),
                                    attention_mask = text_atts[start+i].repeat(k_test,1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        topk_idx = topk_idx.to(device)
        score_matrix_t2v[start+i,topk_idx] = score + topk_sim
    return score_matrix_t2v

@torch.no_grad()
def get_eval_results(model, video_embeds, video_feats, k_test, device='cuda', caption=""):
    tokenizer = model.tokenizer
    caption = caption[:1000]
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
    text_ids = torch.cat([text_input.input_ids], dim=0)
    text_atts = torch.cat([text_input.attention_mask], dim=0)
    text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
    sims_matrix = video_embeds @ text_embed.t()
    sims_matrix = sims_matrix.t()
    score_matrix_t2v = get_t2v_score_matrix(sims_matrix, video_feats, text_ids, text_atts, model, k_test)
    scores_t2v = score_matrix_t2v.cpu().numpy()
    ranks = np.zeros(scores_t2v.shape[0])
    for index, score in enumerate(scores_t2v):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == index)[0][0]
    results = get_metric_results(ranks)
    return results

def get_scores(model, video_embeds, video_feats, k_test, device='cuda', caption="", aggregate_captions=[], update_sims=None):
    tokenizer = model.tokenizer
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
    text_ids = torch.cat([text_input.input_ids], dim=0)
    text_atts = torch.cat([text_input.attention_mask], dim=0)
    text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
    sims_matrix = video_embeds @ text_embed.t()
    sims_matrix = sims_matrix.t()
    if update_sims is not None:
        sims_matrix = sims_matrix * update_sims
        # sims_matrix = torch.clamp(sims_matrix, max=0.999)
    sims_matrix_list = [sims_matrix]

    for aggregate_caption in aggregate_captions:
        text_embed_aggregate = get_single_text_embed(model, aggregate_caption, device='cuda')
        sims_matrix2 = video_embeds @ text_embed_aggregate.t()
        sims_matrix2 = sims_matrix2.t()
        sims_matrix_list.append(sims_matrix2)

    sims_matrix_final = torch.vstack(sims_matrix_list)
    sims_matrix_final = torch.mean(sims_matrix_final, dim=0, keepdim=True)

    score_matrix_t2v = get_t2v_score_matrix(sims_matrix_final, video_feats, text_ids, text_atts, model, k_test)
    scores_t2v = score_matrix_t2v.cpu().numpy()
    score = scores_t2v[0]

    return score

def get_inds_sa(model, video_embeds, video_feats, k_test, device='cuda', caption="", aggregate_captions=[], update_sims=None):
    score = get_scores(model, video_embeds, video_feats, k_test, device=device,
                       caption=caption, aggregate_captions=aggregate_captions, update_sims=update_sims)
    inds = np.argsort(score)[::-1]
    return inds

def get_inds_v2t(model, video_embeds, video_feats, k_test, device='cuda', caption=""):
    tokenizer = model.tokenizer
    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
        device)
    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
    text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:, 0, :]))
    text_ids = torch.cat([text_input.input_ids], dim=0)
    text_atts = torch.cat([text_input.attention_mask], dim=0)
    text_ids[:, 0] = tokenizer.additional_special_tokens_ids[0]
    video_embeds = video_embeds.unsqueeze(dim=0)
    video_feats = video_feats.unsqueeze(dim=0)
    sims_matrix = video_embeds @ text_embed.t()
    score_matrix_v2t = get_v2t_score_matrix(sims_matrix, video_feats, text_ids, text_atts, model, sims_matrix.shape[1])
    scores_v2t = score_matrix_v2t.cpu().numpy()
    score = scores_v2t[0]
    inds = np.argsort(score)[::-1]
    return inds


def get_metric_results(ranks):
    mdR = np.median(ranks + 1)
    # Compute metrics
    vr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    vr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    vr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    vr_mean = (vr1 + vr5 + vr10) / 3
    return vr1, vr5, vr10, vr_mean, mdR