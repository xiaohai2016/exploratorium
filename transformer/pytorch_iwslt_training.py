"""
IWSLT training & related classes
"""
# pylint: disable=W0603
# W0603: use of globals

import torch
import torch.nn as nn

from torchtext import data, datasets
import spacy
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import numpy as np

from pytorch_transformer import make_model, LabelSmoothing, NoamOpt
from pytorch_transformer import SimpleLossCompute, run_epoch, Batch
from pytorch_transformer import MultiGPULossCompute, greedy_decode

class TightBatchingIterator(data.Iterator):
    """
    To contruct tight batching to improve training speed.
    """
    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None):
        super(TightBatchingIterator, self).__init__(dataset, batch_size, \
                sort_key=sort_key, device=device, \
                batch_size_fn=batch_size_fn, train=train, \
                repeat=repeat, shuffle=shuffle, sort=sort, \
                sort_within_batch=sort_within_batch)
        self.batches = None

    def create_batches(self):
        """
        When training, we extract 100 batches from torchtext, sort them by size,
        and then send them as individual batches.
        For non-training, we simply perform a sort inside a batch.
        """
        if self.train:
            def pool(data_in, random_shuffler):
                for macro_batch in data.batch(data_in, self.batch_size * 100):
                    one_batch_iterator = data.batch(sorted(macro_batch, key=self.sort_key), \
                                           self.batch_size, self.batch_size_fn)
                    for one_batch in random_shuffler(list(one_batch_iterator)):
                        yield one_batch
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for one_batch in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(one_batch, key=self.sort_key))

def rebatch(padding_idx, batch):
    "Fix order in torchtext to match ours"
    src, tgt = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, tgt, padding_idx)

# ### the Batch Size Function
# The training utilizes [torchtext package](https://github.com/pytorch/text)
# for accessing datasets and preprocessing. For this purpose and dynamic
# batching, a batch size calculation function is to be provided.

MAX_SRC_IN_BATCH = 0
MAX_TGT_IN_BATCH = 0
def my_batch_size_fn(new, count, _):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global MAX_SRC_IN_BATCH, MAX_TGT_IN_BATCH
    if count == 1:
        MAX_SRC_IN_BATCH = 0
        MAX_TGT_IN_BATCH = 0
    MAX_SRC_IN_BATCH = max(MAX_SRC_IN_BATCH, len(new.src))
    MAX_TGT_IN_BATCH = max(MAX_TGT_IN_BATCH, len(new.trg) + 2)
    # torchtext automatically pads sequences to the maximum sequence length
    src_elements = count * MAX_SRC_IN_BATCH
    tgt_elements = count * MAX_TGT_IN_BATCH
    return max(src_elements, tgt_elements)

class IWSLTTrainer:
    """The training class utilizes CPU/GPU for training"""
    def __init__(self, use_gpu=False, devices=None, from_lang="de", to_lang="en", \
                       batch_size=12000):
        self.use_gpu = use_gpu
        self.devices = devices

        spacy_from = spacy.load(from_lang)
        spacy_to = spacy.load(to_lang)
        tokenizer_from = lambda x: [tok.text for tok in spacy_from.tokenizer(x)]
        tokenizer_to = lambda x: [tok.text for tok in spacy_to.tokenizer(x)]

        bos_word = '<s>'
        eos_word = '</s>'
        blank_word = '<blank>'
        src_field = data.Field(tokenize=tokenizer_from, pad_token=blank_word)
        tgt_field = data.Field(tokenize=tokenizer_to, init_token=bos_word, \
                     eos_token=eos_word, pad_token=blank_word)
        max_len = 100
        self.train, self.val, self.test = datasets.IWSLT.splits(
            exts=('.' + from_lang, '.' + to_lang), fields=(src_field, tgt_field),
            filter_pred=lambda x: len(vars(x)['src']) <= max_len and \
                          len(vars(x)['trg']) <= max_len)

        min_freq = 2
        src_field.build_vocab(self.train.src, min_freq=min_freq)
        tgt_field.build_vocab(self.train.trg, min_freq=min_freq)
        self.pad_idx = tgt_field.vocab.stoi[blank_word]
        self.tgt_field = tgt_field
        self.src_field = src_field

        self.model = make_model(len(src_field.vocab), len(tgt_field.vocab), layer_count=6)
        if use_gpu:
            self.model.cuda()
        self.criterion = LabelSmoothing(size=len(tgt_field.vocab), \
                    padding_idx=self.pad_idx, smoothing=0.1)
        if use_gpu:
            self.criterion.cuda()

        self.train_iter = TightBatchingIterator(self.train, batch_size=batch_size, device=None, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=my_batch_size_fn, train=True)
        self.valid_iter = TightBatchingIterator(self.val, batch_size=batch_size, device=None, \
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), \
                            batch_size_fn=my_batch_size_fn, train=False)
        if use_gpu:
            self.model_par = nn.DataParallel(self.model, device_ids=devices)

        self.model_opt = NoamOpt(self.model.src_embed[0].d_model, 1, 2000, \
            torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    def run_training(self, epoch_count, log_interval=50):
        '''Run training for multiple epochs'''
        for _ in range(epoch_count):
            if self.use_gpu:
                self.model.train()
                run_epoch((rebatch(self.pad_idx, b) for b in self.train_iter), \
                      self.model_par, \
                      MultiGPULossCompute(self.model.generator, self.criterion, \
                                          devices=self.devices, opt=self.model_opt), \
                      log_interval=log_interval)
                self.model.eval()
                loss = run_epoch((rebatch(self.pad_idx, b) for b in self.valid_iter), \
                            self.model_par, \
                            MultiGPULossCompute(self.model.generator, self.criterion, \
                                devices=self.devices, opt=None), \
                            log_interval=log_interval)
                print("Validation loss: ", loss)
                self.run_evaluation(self.model, gpu_devices=self.devices)
            else:
                self.model.train()
                run_epoch((rebatch(self.pad_idx, b) for b in self.train_iter), \
                        self.model, \
                        SimpleLossCompute(self.model.generator, self.criterion, \
                                        opt=self.model_opt), \
                        log_interval=log_interval)
                self.model.eval()
                loss = run_epoch((rebatch(self.pad_idx, b) for b in self.valid_iter), \
                            self.model, \
                            SimpleLossCompute(self.model.generator, self.criterion, opt=None), \
                            log_interval=log_interval)
                print("Validation loss: ", loss)
                self.run_evaluation(self.model)

    def run_evaluation(self, model, gpu_devices=None):
        """
        Compute BLEU score for each test sentence, and the overall average.
        """
        ref_sentences = []
        hyp_sentences = []
        smoothie = SmoothingFunction().method0
        for _, batch in enumerate(self.valid_iter):
            src = batch.src.transpose(0, 1)
            src_mask = (src != self.src_field.vocab.stoi["<blank>"]).unsqueeze(-2)
            out = greedy_decode(model, src, src_mask, gpu_devices=gpu_devices, \
                        max_len=60, start_symbol=self.tgt_field.vocab.stoi["<s>"])
            for hyp in out:
                hypothesis = []
                for j in range(1, hyp.size(0)):
                    sym = self.tgt_field.vocab.itos[hyp[j]]
                    if sym == "</s>":
                        break
                    hypothesis.append(sym)
                hyp_sentences.append(hypothesis)
            tgt = batch.trg.transpose(0, 1)
            for sen in tgt:
                reference = []
                for j in range(1, sen.size(0)):
                    sym = self.tgt_field.vocab.itos[sen.data[j]]
                    if sym == "</s>":
                        break
                    reference.append(sym)
                ref_sentences.append(reference)

        assert len(ref_sentences) == len(hyp_sentences)
        bleu_scores = [sentence_bleu(ref, cand, smoothing_function=smoothie) for \
                       (ref, cand) in zip(ref_sentences, hyp_sentences)]
        max_print = 5
        print_count = 0
        for (score, ref, candidate) in zip(bleu_scores, ref_sentences, hyp_sentences):
            print("Score: {0:.2f},\n\t ref:{1}\n\tpred:{2}".format(
                score, ref, candidate))
            print_count += 1
            if print_count >= max_print:
                break
        print("BLEU scores: mean - {:.2f}, max - {:.2f}, min - {:.2f}, std - {:.2f}".format( \
            np.mean(bleu_scores) * 100, \
            np.amax(bleu_scores) * 100, \
            np.amin(bleu_scores) * 100, \
            np.std(bleu_scores) * 100, \
            ))
