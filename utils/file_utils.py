from utils.config import  save_result_dir
import time
import os
def save_vocab(vocab_path, vocab):
    with open(vocab_path) as f:
        for i in vocab:
            f.write(i)


def save_dict(file_path, dict_data):
    """
    保存词典
    :param file_path:
    :param dict_data:
    :return:
    """
    with open(file_path, 'w', encoding = "utf-8") as f:
        for k,v in dict_data.items():
            f.write("{}\t{}\n".format(k,v))



def get_result_filename(batch_size, epochs, max_length_inp, embedding_dim, commit=''):
    """
    获取时间
    :return:
    """
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = now_time + '_batch_size_{}_epochs_{}_max_length_inp_{}_embedding_dim_{}{}.csv'.format(batch_size, epochs,
                                                                                                   max_length_inp,
                                                                                                   embedding_dim,
                                                                                                   commit)
    result_save_path = os.path.join(save_result_dir, filename)
    return result_save_path