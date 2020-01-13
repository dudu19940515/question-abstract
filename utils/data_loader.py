import re
import jieba
import pandas as pd
import numpy as np
import codecs
from utils.file_utils import save_vocab, save_dict
from utils.multi_proc_utils import parallelize
from utils.config import stop_word_path, train_data_path, test_data_path
from utils.config import train_seg_path, test_seg_path, merger_seg_path, user_dict
from utils.config import train_x_pad_path, train_y_pad_path, test_x_pad_path, wv_train_epochs, model_path_trained
from utils.config import train_x_path, train_y_path, test_x_path,embedding_dim
from utils.config import embedding_matrix_path, vocab_path,reverse_vocab_path, word2vec_type
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.models.fasttext import FastText
from sklearn.model_selection import train_test_split
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_data(max_enc_len=300, max_dec_len=50):
    '''
    加载处理好的数据集
    :return:
    '''
    train_x = np.load(train_x_path+'.npy')
    train_y = np.load(train_y_path+'.npy')
    test_x = np.load(test_x_path+'.npy')
    train_x = train_x[:, :max_enc_len]
    train_y = train_y[:, :max_dec_len]
    test_x = test_x[:, :max_enc_len]
    #trainx = train_x.astype("float64")
    #train_y = train_y.astype("float64")
    #test_x = test_x.astype("float64")
    return train_x, train_y, test_x

def load_train_dataset(enc_max_len = 300, dec_max_len = 50):
    train_x = np.load(train_x_path+'.npy')
    train_y = np.load(train_y_path+'.npy')

    train_x = train_x[:,:enc_max_len]
    train_y = train_y[:,:dec_max_len]
    return train_x, train_y
def load_test_dataset(max_enc_len=200):
    """
    :return: 加载处理好的数据集
    """
    test_X = np.load(test_x_path + '.npy')
    test_X = test_X[:, :max_enc_len]
    return test_X



if __name__ == '__main__':
    # 数据集批量处理
    #data_generate(train_data_path, test_data_path)
    build_dataset(train_data_path, test_data_path)