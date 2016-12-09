import sys
import os
import copy
import re
import argparse
from operator import itemgetter
from collections import Counter
import json
import h5py
import numpy as np


def encode_txt(samples, max_length, wtoi):
    N = len(samples)
    question_id = np.zeros((N,), dtype='uint32')
    txt = np.zeros((N, max_length), dtype='uint32')

    unk_cnt = 0
    trun_cnt = 0
    unk_idx = wtoi.get('**unknown**')
    for i,sample in enumerate(samples):
        question_id[i] = sample['question_id']

        words = sample['question'].split()
        nword = len(words)
        for j,w in enumerate(words):
            if j >= max_length:
                trun_cnt += 1
                break
            txt[i][max_length-nword+j] = wtoi.get(w, unk_idx) # right alignment
            unk_cnt += (0 if w in wtoi else 1)

    print('[Info] Truncated txt count: {}'.format(trun_cnt))
    print('[Info] Unknown word count: {}'.format(unk_cnt))
    return txt, question_id


def encode_answer(samples, atoi):
    answer = np.zeros(len(samples), dtype='uint32')
    for i,sample in enumerate(samples):
        answer[i] = atoi.get(sample['answer'])
    return answer


def encode_MC(samples, atoi):
    MC = np.zeros((len(samples),18), dtype='uint32')
    for i,s in enumerate(samples):
        for j,m in enumerate(s['multiple_choices']):
            MC[i,j] = atoi.get(m, 0)
    return MC


def index_img_fea(splits, samples, opt):
    img_fea_file = []
    img_ids = []

    for split in splits:
        if split == 'test-dev2015':
            split = 'test2015'
        fea_fname = '{}/VQA_{}_{}'.format(
                opt.imfea_path, split, opt.imfea_name)
        if not opt.imfea_no_norm:
            norm_fea_fname = '{}_norm.npy'.format(fea_fname)

            # Normalize image feature
            if not os.path.exists(norm_fea_fname):
                fea_fname += '.npy'
                print('[Info] Normalize {}'.format(fea_fname))
                fea = np.load(fea_fname)
                nm = np.linalg.norm(fea, axis=1)
                nm = np.repeat(nm[:, np.newaxis], fea.shape[1], axis=1)
                fea /= nm
                np.save(norm_fea_fname, fea)

            img_fea_file.append(norm_fea_fname)
        else:
            img_fea_file.append(fea_fname+'.npy')

        path_fname = '{}/VQA_{}_image-path.txt'.format(opt.imfea_path, split)
        print('[Loading] {}'.format(path_fname))
        with open(path_fname) as f:
            img_paths = f.readlines()
        get_id = lambda p: int(p.rsplit(' ', 1)[0].rsplit('.', 1)[0].rsplit('_', 1)[1])
        img_ids.extend(map(get_id, img_paths))

    id_to_pos = {img_id:pos+1 for pos,img_id in enumerate(img_ids)}
    assert(len(id_to_pos) == len(img_ids))

    img_pos = np.zeros(len(samples), dtype='uint32')
    for i,sample in enumerate(samples):
        img_pos[i] = id_to_pos.get(sample['image_id'])

    return img_pos, img_fea_file


def load_data(data_dir, splits):
    samples = []
    for split in splits:
        file_name = '{}/{}.json'.format(data_dir, split)
        print('[Loading] {}'.format(file_name))
        with open(file_name) as f:
            samples.extend(json.load(f))
    return samples


def main(opt):
    if opt.split == 1:
        train_split = ('train2014',)
        test_split = ('val2014',)
    elif opt.split == 2:
        train_split = ('train2014', 'val2014')
        test_split = ('test-dev2015',)
    else:
        train_split = ('train2014', 'val2014')
        test_split = ('test2015',)
    print('[Info] train split: {}'.format(train_split))
    print('[Info] test split: {}'.format(test_split))

    # Load training data
    train_sample = load_data(opt.data_dir, train_split)
    test_sample = load_data(opt.data_dir, test_split)

    result = {}

    # Get top answers
    answer = map(itemgetter('answer'), train_sample)
    top_answer = map(itemgetter(0), Counter(answer).most_common(opt.ans_topk))
    print('[Info] Use top {} answers'.format(len(top_answer)))
    itoa = top_answer
    atoi = {a: i+1 for i,a in enumerate(itoa)}
    assert(len(atoi) == len(itoa))
    result['atoi'] = atoi
    result['itoa'] = itoa

    # Filter unknown answer in training set
    count = len(train_sample)
    train_sample = filter(lambda s: s['answer'] in atoi, train_sample)
    result['train_unk_ans_cnt'] = count - len(train_sample)
    print('[Info] Unknown answer count in training set: {}({:%})'
            .format(result['train_unk_ans_cnt'], 1-1.*len(train_sample)/count))
    result['train_sample'] = train_sample
    result['test_sample'] = test_sample

    # Build vocabulary
    word_counter = Counter()
    for s in train_sample:
        word_counter.update(s['question'].split())
    word_cnt = word_counter.most_common()
    word_cnt.reverse()
    for i,wc in enumerate(word_cnt):
        if wc[1] >= opt.word_min_freq:
            vocab = map(itemgetter(0), word_cnt[i:])
            break
    res_cnt = len(vocab)
    all_cnt = len(word_cnt)
    print('[Info] Reserved words count: {}/{}({:%})'.format(res_cnt, all_cnt, 1.*res_cnt/all_cnt))
    
    # Add **unkown** to vocabulary
    vocab.append('**unknown**')
    itow = vocab
    wtoi = {w: i+1 for i,w in enumerate(itow)}
    assert(len(wtoi) == len(itow))
    result['wtoi'] = wtoi
    result['itow'] = itow

    # Index image feature
    train_img_pos, train_img_fea_file = index_img_fea(train_split, train_sample, opt)
    test_img_pos, test_img_fea_file = index_img_fea(test_split, test_sample, opt)
    result['train_img_fea_file'] = train_img_fea_file
    result['test_img_fea_file'] = test_img_fea_file

    # Encode txt
    print('[Info] Training Set')
    train_txt, train_ques_id = encode_txt(train_sample, opt.max_txt_len, wtoi)
    train_ans = encode_answer(train_sample, atoi)
    print('[Info] Test Set')
    test_txt, test_ques_id = encode_txt(test_sample, opt.max_txt_len, wtoi)
    test_MC = encode_MC(test_sample, atoi)

    # Compose save file name
    a1000 = opt.ans_topk/1000
    a100 = opt.ans_topk%1000/100
    a100 = '' if a100 == 0 else a100
    out_fname = 'data_s{}_{}_a{}k{}_wmf{}_mtl{}'.format(
            opt.split, opt.imfea_name, a1000, a100, 
            opt.word_min_freq, opt.max_txt_len)

    # Save
    json_fname = '{}/{}.json'.format(opt.data_dir, out_fname)
    print('[Storing] {}'.format(json_fname))
    with open(json_fname, 'w') as f:
        json.dump(result, f)

    h5_fname = '{}/{}.h5'.format(opt.data_dir, out_fname)
    print('[Storing] {}'.format(h5_fname))
    with h5py.File(h5_fname, 'w') as f:
        f.create_dataset('train_ques_id', dtype='uint32', data=train_ques_id)
        f.create_dataset('train_img_pos', dtype='uint32', data=train_img_pos)
        f.create_dataset('train_txt', dtype='uint32', data=train_txt)
        f.create_dataset('train_ans', dtype='uint32', data=train_ans)
        f.create_dataset('test_ques_id', dtype='uint32', data=test_ques_id)
        f.create_dataset('test_img_pos', dtype='uint32', data=test_img_pos)
        f.create_dataset('test_txt', dtype='uint32', data=test_txt)
        f.create_dataset('test_MC', dtype='uint32', data=test_MC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare data to train.")

    parser.add_argument('--data_dir', default='data', help='where the row data stored')
    parser.add_argument('--split', default=1, type=int, choices=[1,2,3], help='split setting (1: train2014/val2014; 2: train2014+val2014/test-dev2015; 3: train2014+val2014/test2015)')
    parser.add_argument('--ans_topk', default=1000, type=int, help='number of top answer for classification')
    parser.add_argument('--word_min_freq', default=1, type=int, help='words that occur less than this number will be discarded')
    parser.add_argument('--max_txt_len', default=26, type=int, help='maximum length of the txt input')

    parser.add_argument('--imfea_path', default='data/feature', help='where the image feature stored')
    parser.add_argument('--imfea_name', default='VGG19-4096', help='feature name in feature file name')
    parser.add_argument('--imfea_no_norm', action='store_true', help='do not normalize image feature')

    opt = parser.parse_args()
    print('[Info] parsed input parameters:')
    print(json.dumps(vars(opt), indent = 2))
    main(opt)
