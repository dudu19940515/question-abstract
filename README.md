# question-abstract
百度问答摘要比赛记录 **1月12日**
## 第一版解答为一般seq-seq模型，采用一般机器翻译模型：
encoder设置为双向gru,decoder为单向gru，引入attention机制

## 第二版引入pointer-network
encoder为双向lstm, decoder 为单向lstm，inference采用beam search

