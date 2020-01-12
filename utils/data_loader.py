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
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 自定义词表
jieba.load_userdict(user_dict)

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


def build_dataset(train_data_path, test_data_path, word2vec_type=True):
    '''
    构建数据集

    :param train_data_path:
    :param test_data_path:
    :param w2v_model_trained_path: 如果有已经训练好的词向量
    :return:
    '''
    # 1.加载数据
    train_df, test_df = load_dataset(train_data_path, test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))
    # 2.空值清洗
    train_df.dropna(subset=['Report'], inplace=True)
    #test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    test_df.fillna('', inplace=True)
    # 3.并行化处理，多线程处理

    train_df = parallelize(train_df, data_frame_proc)
    test_df = parallelize(test_df, data_frame_proc)
    # 4. 保存处理完成的数据
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)

    # 5. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)

    train_df = train_df.drop(['merged'], axis=1)
    test_df = test_df.drop(['merged'], axis=1)

     # 6. 保存合并数据
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)

    merged_df.to_csv(merger_seg_path, index=None, header=False)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                               len(merged_df)))
    #7.词向量训练
    print('start build w2v model')

    if word2vec_type:
        wv_model = Word2Vec(LineSentence(merger_seg_path), size=embedding_dim,
                            negative=5,
                            workers=8,
                            iter=wv_train_epochs,
                            window=3,
                            min_count=5)
    else:
        wv_model = FastText(LineSentence(merger_seg_path), workers=8, min_count=5, size=300, window = 3,iter=wv_train_epochs)

    # 8. 分离数据和标签
    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    # 9. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab
    # 训练集X处理
    # 获取适当的最大长度
    train_x_max_len = get_max_len(train_df['X'])
    test_X_max_len = get_max_len(test_df['X'])
    X_max_len = max(train_x_max_len, test_X_max_len)

    print("training sequence length is: ", X_max_len)
    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, X_max_len, vocab))

    # 训练集Y处理
    # 获取适当的最大长度
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))
    print("report sequence length is: ", train_y_max_len)

    # 10. 保存pad oov处理后的,数据和标签
    train_df['X'].to_csv(train_x_pad_path, index=None, header=False)
    train_df['Y'].to_csv(train_y_pad_path, index=None, header=False)
    test_df['X'].to_csv(test_x_pad_path, index=None, header=False)

    # 11. 词向量再次训练
    print('start retrain w2v model')
    wv_model.build_vocab(LineSentence(train_x_pad_path), update=True)
    wv_model.train(LineSentence(train_x_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)
    print('1/3')
    wv_model.build_vocab(LineSentence(train_y_pad_path), update=True)
    wv_model.train(LineSentence(train_y_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)
    print('2/3')
    wv_model.build_vocab(LineSentence(test_x_pad_path), update=True)
    wv_model.train(LineSentence(test_x_pad_path), epochs=wv_train_epochs, total_examples=wv_model.corpus_count)

    # 保存词向量模型
    wv_model.save(model_path_trained)
    print('finish retrain w2v model')
    print('final w2v_model has vocabulary of ', len(wv_model.wv.vocab))

    #12 更新词典
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index: word for index, word in enumerate(wv_model.wv.index2word)}
    #保存词典
    save_dict(vocab_path,vocab)
    save_dict(reverse_vocab_path, reverse_vocab)

    #13 保存词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    np.save(embedding_matrix_path, embedding_matrix)

    #14 将数据集转换为索引
    train_ids_x = train_df['X'].apply(lambda x: transform_data(x, vocab))
    train_ids_y = train_df['Y'].apply(lambda x: transform_data(x,vocab))
    test_ids_x = test_df['X'].apply(lambda x: transform_data(x, vocab))

    # 15. 数据转换成numpy数组
    # 将索引列表转换成矩阵 [32800, 403, 986, 246, 231] --> array([[32800,   403,   986 ]]
    train_X = np.array(train_ids_x.tolist())
    train_Y = np.array(train_ids_y.tolist())
    test_X = np.array(test_ids_x.tolist())

    # 保存数据
    np.save(train_x_path, train_X)
    np.save(train_y_path, train_Y)
    np.save(test_x_path, test_X)

    return train_X, train_Y, test_X

def preprocess_sentence(sentence,max_len, vocab):
    sentence = sentence_proc(sentence)
    sentence = pad_proc(sentence, max_len-2, vocab)
    sentence = transform_data(sentence, vocab)

    return np.array([sentence])


def transform_data(sentence, vocab):
    '''
    将字词转换为id
    :param sentence:
    :param vocab:
    :return:
    '''
    unk_id = vocab['<UNK>']
    words = sentence.split(" ")
    ids = [vocab[word] if word in vocab else unk_id for word in words]
    return ids

def data_generate(train_data_path,test_data_path):
    '''
    第一版数据处理#数据加载与预处理
    :param train_data_path:
    :param test_data_path:
    :return:
    '''
    #1.加载数据
    train_df,test_df = load_dataset(train_data_path, test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))
    #2.空值清洗
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how="any", inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    #3.并行化处理，多线程处理
    train_df = parallelize(train_df, data_frame_proc)
    test_df = parallelize(test_df, data_frame_proc)
    # 4. 保存处理完成的数据
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)

    #5. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                          len(merged_df)))
    # 6. 保存合并数据
    merged_df.to_csv(merger_seg_path, index=None, header=True)

    return train_df, test_df, merged_df


def load_dataset(train_data_path, test_data_path):
    '''
    数据数据集
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return:
    '''
    # 读取数据集
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    return train_data, test_data


def clean_sentence(sentence):
    '''
    清洗数据
    :param sentence:
    :return:
    '''
    if isinstance(sentence, str):
        return re.sub(
            #r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            r'[\s+\-\/\[\]\{\}_$%^*(+\"\')]+|[+——()【】“”~@#￥%……&*（）]+|你好|您好',
            '', sentence)
    else:
        return ' '


def load_stop_words(stop_word_path):
    '''
    导入停用词
    :param stop_word_path:
    :return:
    '''
    with open(stop_word_path, 'r', encoding='utf-8') as file:
        stopwords = file.readlines()
        stopwords = [stopword.strip() for stopword in stopwords] #清除换行号
    print('stop words size {}'.format(len(stopwords)))
    return stopwords

stop_words = load_stop_words(stop_word_path)
def filter_stopwords(sentence):
    '''
    过滤停用词
    words: 已经切好的词的list
    '''
    words = sentence.split(' ')
    # 去掉多余空字符
    words = [word for word in words if word]
    return [word for word in words if word not in stop_words]

def seg_proc(sentence):
    '''

    '''
    tokens = sentence.split('|')
    result = []
    for t in tokens:
        result.append(cut_sentence(t))
    return ' | '.join(result)

def cut_sentence(line):
    # 切词，默认精确模式，全模式cut参数cut_all=True
    tokens = jieba.cut(line)
    return ' '.join(tokens)

def sentence_proc(sentence):
    '''
    批量预处理
    :param sentence:
    :return:
    '''
    cleaned_sentence = clean_sentence(sentence)
    sentence_cut = seg_proc(cleaned_sentence)
    filter_sentence = filter_stopwords(sentence_cut)

    return " ".join(filter_sentence)

def data_frame_proc(df):
    """
    dataframe 的预处理
    :param df:
    :return:
    """
    for column in ['Brand','Model','Question','Dialogue']:
        df[column] = df[column].apply(sentence_proc)
    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(sentence_proc)
    return df

def pad_proc(sentence, max_len, vocab):
    '''
    < start > < end > < pad > < unk >
    '''
    # 0.按空格统计切分出词
    words = sentence.strip().split(' ')
    # 1. 截取规定长度的词数
    words = words[:max_len]
    # 2. 填充< unk > ,判断是否在vocab中, 不在填充 < unk >
    sentence = [word if word in vocab else '<UNK>' for word in words]
    # 3. 填充< start > < end >
    sentence = ['<START>'] + sentence + ['<STOP>']
    # 4. 判断长度，填充　< pad >
    sentence = sentence + ['<PAD>'] * (max_len  - len(words))
    return ' '.join(sentence)

def get_max_len(data):
    """
    获得合适的最大长度值
    :param data: 待统计的数据  train_df['Question']
    :return: 最大长度值
    """
    max_lens = data.apply(lambda x: x.count(' ')+1)
    return int(np.mean(max_lens) + 2 * np.std(max_lens))
if __name__ == '__main__':
    # 数据集批量处理
    #data_generate(train_data_path, test_data_path)
    build_dataset(train_data_path, test_data_path)