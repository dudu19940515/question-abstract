# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent

# 训练数据路径
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
# 测试数据路径
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
# 停用词路径
stop_word_path = os.path.join(root, 'data', 'stopwords/stopwords.txt')

# 自定义切词表
user_dict = os.path.join(root, 'data', 'user_dict3.txt')

# 预处理后的训练数据
train_seg_path = os.path.join(root, 'data', 'train_seg_data.csv')
# 预处理后的测试数据
test_seg_path = os.path.join(root, 'data', 'test_seg_data.csv')
# 合并训练集测试集数据
merger_seg_path = os.path.join(root, 'data', 'merged_train_test_seg_data.csv')

# 2. pad oov处理后的数据
train_x_pad_path = os.path.join(root, 'data', 'train_X_pad_data.csv')
train_y_pad_path = os.path.join(root, 'data', 'train_Y_pad_data.csv')
test_x_pad_path = os.path.join(root, 'data', 'test_X_pad_data.csv')

# 3. numpy 转换后的数据
train_x_path = os.path.join(root, 'data', 'train_data_X')
train_y_path = os.path.join(root, 'data', 'train_data_Y')
test_x_path = os.path.join(root, 'data', 'test_data_X')


#词表的路径
vocab_path = os.path.join(root, 'data', 'wv', 'vocab.txt')
reverse_vocab_path = os.path.join(root, 'data', 'wv', 'reverse_vocab.txt')
#word2vec模型的路径
model_path = os.path.join(root, 'data', 'wv', 'word2vec.model')
#完全填充训练后的模型
model_path_trained = os.path.join(root,'data', 'wv', 'word2vec_trained.model')
#fast_text模型的路径
model_path_fasttext = os.path.join(root,'data','wv','fast_text.model')
#embedding矩阵的路径
embedding_matrix_path = os.path.join(root, 'data', 'wv','embedding_matrix')

# 模型保存文件夹
checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_mask_loss_dim300_seq')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(root, 'result')
# 词向量训练轮数
wv_train_epochs = 10
word2vec_type = True #是否采用word2vec 否则采用 fasttext

# 词向量维度
embedding_dim = 300