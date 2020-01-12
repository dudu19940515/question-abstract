from gensim.models.word2vec import LineSentence
from gensim.models import word2vec
from gensim.models.fasttext import FastText
import gensim
import numpy as np
from utils.config import  merger_seg_path, model_path, model_path_fasttext, embedding_path,vocab_path
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def w2v_training(merger_seg_path, min_count = 5, embeding_size = 300):
    '''
    word2vecx 训练
    :param merger_seg_path:
    :param min_count:
    :param embeding_size:
    :return:
    '''
    #1.模型训练
    model = word2vec.Word2Vec(LineSentence(merger_seg_path), workers=8, \
                              min_count=min_count, size=embeding_size)
    #2.模型保存
    model.save(model_path)
    print("奇瑞最相近的词汇：\n", model.wv.most_similar(['奇瑞'], topn=10))

def fast_text_training(merger_seg_path, model_path, min_count = 5, embeding_size =200):
    '''
    使用fast_text方法训练
    :param merger_seg_path:
    :param min_count:
    :param embeding_size:
    :return:
    '''
    model = FastText(sentences=LineSentence(merger_seg_path), workers=8, min_count=5, size=200)
    model.save(model_path)
    print("奇瑞最相近的词汇：\n", model.wv.most_similar(['奇瑞'], topn=10))

def build_embeding_matrix(model_path, embeding_path):
    '''
    正规保存方法
    :param model_path:
    :param embeding_path:
    :return:
    '''
    model = word2vec.Word2Vec.load(model_path)
    embedding_matrix = model.wv.vectors
    np.savetxt(embeding_path, embedding_matrix, fmt='%0.8f')
    print(embedding_matrix.shape)
    print('embedding matrix extracted')


def build_embedding_matrix_forhomework(model_path,vocab_path,embedding_path):
    """
    作业2用的方法
    根据词表建立词嵌入矩阵
    :param model_path:
    :param vocab_path:
    :return:
    """
    #1.导入词表
    vocab = []
    with open(vocab_path, 'r') as f:
        for word in f.readlines():
            vocab.append(word.strip())
    #2.导入word2vec模型
    model = word2vec.Word2Vec.load(model_path)
    #3.构建词嵌入矩阵
    embedding_matrix = []
    for index, word in enumerate(vocab):
        try:
            embedding_matrix.append(model.wv[word])
        except Exception as e:
            print('词表没这个词')
            print(e)

    return embedding_matrix

if __name__=="__main__":
    #w2v_training(merger_seg_path)
    #embedding_matrix = build_embedding_matrix(model_path,vocab_path,embedding_path)
    fast_text_training(merger_seg_path,model_path_fasttext )
    build_embeding_matrix(model_path_fasttext, embedding_path)


