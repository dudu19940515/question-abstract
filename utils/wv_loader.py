import numpy as np
from utils.config import embedding_matrix_path,vocab_path
import codecs
from gensim.models.word2vec import LineSentence, Word2Vec
# 引入日志配置
import logging

def load_word2vec_file(save_wv_model_path):
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    embedding_matrix = wv_model.wv.vectors
    return embedding_matrix

def get_vocab(save_wv_model_path):
    # 保存词向量模型
    wv_model = Word2Vec.load(save_wv_model_path)
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    return vocab, reverse_vocab

def load_vocab(file_path):
    """
    导入词表
    """
    vocab = {}
    reverse_vocab = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word, index = line.strip().split('\t')
            index = int(index)
            vocab[word] = index
            reverse_vocab[index] = word
    return vocab, reverse_vocab


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'

    def __init__(self, vocab_file, Vocab_max_size = None):
        self.word2id, self.id2word = self.load_vocab(vocab_file, Vocab_max_size)
        self.count = len(self.vocab)

    @staticmethod
    def load_vocab(vocab_file, Vocab_max_size = None):
        vocab = {}
        reverse_vocab = {}

        with open(vocab_file, 'r', 'utf-8') as f:
            for line in f.readlines():
                word,index = line.strip().split('\t')
                index = int(index)

                if Vocab_max_size and Vocab_max_size<index:
                    print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                        vocab_max_size, index))
                    break
                vocab[word] = index
                reverse_vocab[index] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return self.word2id[UNKNOWN_TOKEN]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count




def load_embedding_matrix():

    return np.load(embedding_matrix_path + '.npy')