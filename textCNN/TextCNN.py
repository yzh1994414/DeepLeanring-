""""
Topic : Multi-classification Text_CNN module
Author : Yang Zhenghong
Date : 2019.7.11
""""



""""
AdamW Optimizer 
weight_decay = 0.08
"""
class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/4)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            # decoupled weight decay (2/4)
            self.wd = K.variable(weight_decay, name='weight_decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd  # decoupled weight decay (3/4)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            # decoupled weight decay (4/4)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * wd * p

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Text_CNN(embedding_matrix, maxlen, embed_size = 128, filter_sizes = [1], filter_nums = [128] ,weight_decay = 0.08):
"""
Text_CNN model.
  
Parameters:
  embedding_matrix - pretrained word vector matrix
  maxlen - the max length of sentence 
  embed_size - the size of word vector
  filter_sizes - the size of conv1d filter
  filter_nums - the nums of conv1d filter correspond to filter_sizes
  
Returns:
  The Text_CNN model.
  
Raises:
  None
"""
    model = Sequential()
    seq = Input(shape=[maxlen])
    #Embedding layers
    emb = Embedding(len(embedding_matrix),       #表示文本数据中词汇的取值可能数,从语料库之中保留多少个单词。 因为Keras需要预留一个全零层， 所以+1
               embed_size,              # 嵌入单词的向量空间的大小。它为每个单词定义了这个层的输出向量的大小
               weights=[embedding_matrix],   #构建一个[num_words, EMBEDDING_DIM]的矩阵,然后遍历word_index，将word在W2V模型之中对应vector复制                                        过来。换个方式说：embedding_matrix 是原始W2V的子集，排列顺序按照Tokenizer在fit之后的词顺序。作为权                                      重喂给Embedding Layer
               input_length=maxlen,        # 输入序列的长度，也就是一次输入带有的词汇个数
               trainable=True            # 我们设置 trainable = False，代表词向量不作为参数进行更新
               )(seq)

    convs = []
    for fsz,fnums in zip(filter_sizes,filter_nums):
        conv1 = Conv1D(filters=fnums,kernel_size=fsz,activation='relu')(emb)
        pool1 = MaxPooling1D(int(conv1.shape[1]))(conv1)
        pool1 = Flatten()(pool1)
        convs.append(pool1)
    if len(convs)>1:
        merge = concatenate(convs,axis=1)
        out = Dropout(0.2)(merge)
    else :
        out = Dropout(0.2)(convs[0])
        
    output = Dense(64,activation='relu')(out)

    output = Dense(units=6,activation='sigmoid')(output)

    model = Model([seq],output)
    model.compile(loss='categorical_crossentropy', optimizer=AdamW(weight_decay=weight_decay,),metrics=['accuracy'])
    model.summary()
    return model
    
