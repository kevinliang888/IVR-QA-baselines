import torch
from interactive.utils import get_conjunctions
import pdb

def generate_object_caption(video_frame, model_vqa):
    num_frames = video_frame.shape[0]
    question1 = 'What is the object ?'
    answer1 = model_vqa.generate(video_frame, question1, train=False, inference='generate', num_frames=num_frames)
    question2 = f'What color is the {answer1[0]} ?'
    answer2 = model_vqa.generate(video_frame, question2, train=False, inference='generate', num_frames=num_frames)
    question3 = f'Where is the {answer2[0]} {answer1[0]} ?'
    answer3 = model_vqa.generate(video_frame, question3, train=False, inference='generate', num_frames=num_frames)
    new_caption = f'{answer2[0]} {answer1[0]} is {answer3[0]}'
    return new_caption

def generate_caption_vqa(video_frame, origin_caption, model_vqa):
    split_captions = origin_caption.split(' ')
    num_frames = video_frame.shape[0]

    human_captions = ['person', 'people', 'man', 'men', 'woman', 'women', 'girl', 'girls', 'boy',
                      'boys', 'child', 'children', 'male', 'female', 'lady', 'family', 'models']
    singular_human_captions = ['person', 'man', 'woman', 'girl', 'boy', 'child', 'male', 'female', 'lady', 'family']

    for human_caption in human_captions:
        if human_caption in split_captions:
            is_human = 1
            break
        else:
            is_human = 0
    verb = get_conjunctions(singular_human_captions, human_caption)

    if is_human:
        question1 = f'What {verb} the {human_caption} doing ?'
        answer1 = model_vqa.generate(video_frame, question1, train=False, inference='generate', num_frames=num_frames)
        question2 = f'Where {verb} the {human_caption} {answer1[0]} ?'
        answer2 = model_vqa.generate(video_frame, question2, train=False, inference='generate', num_frames=num_frames)
        if 'no' in answer2[0]:
            new_caption = f'{human_caption} {verb} {answer1[0]}'
        else:
            new_caption = f'{human_caption} {verb} {answer1[0]} {answer2[0]}'
    else:
        question = 'Is this cartoon ?'
        answer = model_vqa.generate(video_frame, question, train=False, inference='generate', num_frames=num_frames)
        if answer[0] == 'yes':
            question1 = 'What is the character ?'
            answer1 = model_vqa.generate(video_frame, question1, train=False, inference='generate', num_frames=num_frames)
            question2 = f'What is the {answer[0]} doing ?'
            answer2 = model_vqa.generate(video_frame, question2, train=False, inference='generate', num_frames=num_frames)
            new_caption = f'{answer1[0]} is {answer2[0]}'
        else:
            question = 'Is there any animal ?'
            answer = model_vqa.generate(video_frame, question, train=False, inference='generate', num_frames=num_frames)
            if answer[0] == 'yes':
                question1 = 'What is the animal ?'
                answer1 = model_vqa.generate(video_frame, question1, train=False, inference='generate', num_frames=num_frames)
                question2 = f'What is the {answer1[0]} doing ?'
                answer2 = model_vqa.generate(video_frame, question2, train=False, inference='generate', num_frames=num_frames)
                question3 = f'Where is the {answer1[0]} ?'
                answer3 = model_vqa.generate(video_frame, question3, train=False, inference='generate', num_frames=num_frames)
                new_caption = f'{answer1[0]} is {answer2[0]} {answer3[0]}'
            else:
                new_caption = generate_object_caption(video_frame, model_vqa)
    return new_caption


@torch.no_grad()
def generate_question_auto(cap_decoder, cap_config, top_k_video_feats, original_query, t0_tokenizer, t0_model, use_caption=False):
    if use_caption:
        captions = cap_decoder.generate_from_image_embeds(top_k_video_feats, sample=False,
                                                          num_beams=cap_config['num_beams'],
                                                          max_length=cap_config['max_length'],
                                                          min_length=cap_config['min_length'],
                                                          repetition_penalty=1.1)
        separator = ', '
        all_caps = separator.join(captions)
        q1 = f"Suppose you are given the following video descriptions: \"{all_caps}\". " \
             f"What question would you ask to help you uniquely identify the video described as follows: \"{original_query}\" ?"
    else:
        q1 = f"Suppose you are given the following video descriptions: \"{original_query}\". " \
             f"What question would you ask to help you uniquely identify the video?"
    inputs = t0_tokenizer.encode(q1, return_tensors="pt")
    inputs = inputs.to(t0_model.device)
    outputs = t0_model.generate(inputs)
    question1 = t0_tokenizer.decode(outputs[0]).replace('<pad> ', '').replace('</s>', '')

    return question1

def generate_caption_vqa_auto(video_frame, model_vqa, question):
    num_frames = video_frame.shape[0]
    question = question.lower()
    answer = model_vqa.generate(video_frame, question, train=False, inference='generate', num_frames=num_frames)

    question = question.strip('?')
    split_qs = question.split(' ')
    verb_list = ["is", "are", "'s"]
    caption = ""
    try:
        if split_qs[0] == "what" and (split_qs[1] == "does" or split_qs[1] == "did" or split_qs[1] == "do"):
            new_list = split_qs[split_qs.index(split_qs[1]) + 1:]
            new_list[new_list.index("do")] = answer[0]
            caption = ' '.join(new_list)
        elif split_qs[0] == "what":
            found = False
            for verb in verb_list:
                if verb in split_qs:
                    found = True
                    break
            if found:
                if "doing" in split_qs:
                    rest = ' '.join(split_qs[split_qs.index(verb) + 1:])
                    caption = rest.replace("doing", answer[0])
                elif "'s" in split_qs:
                    rest = ' '.join(split_qs[split_qs.index("'s") + 1:])
                    caption = f'{answer[0]} {rest}'
                elif verb in split_qs:
                    rest = ' '.join(split_qs[split_qs.index(verb) + 1:])
                    caption = f'{rest} {verb} {answer[0]}'
        elif split_qs[0] == "how" and split_qs[1] == "many":
            if "there" in split_qs:
                split_qs.remove("there")
            rest = ' '.join(split_qs[2:])
            if int(answer[0]) == 0:
                caption = f'no {rest}'
            elif int(answer[0]) == 1:
                caption = f'a {rest}'
            elif int(answer[0]) > 10:
                caption = f'many {rest}'
            else:
                caption = f'a few {rest}'
        elif split_qs[0] == "who":
            rest = ' '.join(split_qs[1:])
            caption = f'{answer[0]} {rest}'
        elif split_qs[0] == "where":
            singular = "is" in split_qs
            verb = "is" if singular else "are"
            rest = ' '.join(split_qs[2:])
            caption = f'{rest} {verb} {answer[0]}'
    except:
        pass
    return caption

def generate_augment_caption(video_frame, model_vqa, origin_caption, num_segment=2, ask_object=True, ask_regular=True):
    num_frames = video_frame.shape[0]
    step = num_frames // num_segment
    generate_cap = []
    for k in range(0, num_frames, step):
        if ask_regular:
            caption = generate_caption_vqa(video_frame[k:k+step], origin_caption, model_vqa)
            generate_cap.append(caption)
        if ask_object:
            caption = generate_object_caption(video_frame[k:k+step], model_vqa)
            generate_cap.append(caption)
    return generate_cap

def generate_augment_caption_auto(video_frame, model_vqa, question, num_segment=2):
    num_frames = video_frame.shape[0]
    step = num_frames // num_segment
    generate_cap = []
    for k in range(0, num_frames, step):
        caption = generate_caption_vqa_auto(video_frame[k:k + step], model_vqa, question)
        generate_cap.append(caption)
    return generate_cap