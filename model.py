import tensorflow as tf
from keras import initializers
from spektral.layers import GCNConv
from keras.layers import LSTM, Bidirectional, Dense, Dropout, GRU


class TSModel(tf.keras.models.Model):
    def __init__(self):
        super(TSModel, self).__init__()
        self.bilstm = Bidirectional(LSTM(128, return_sequences=False))
        self.fc = Dense(2, activation="softmax")
        self.dropout = Dropout(0.5)

    @tf.function
    def call(self, x, training=False):
        return self.fc(self.dropout(self.bilstm(x), training=training))


class NaiveMultimodalModel(tf.keras.models.Model):
    def __init__(self):
        super(NaiveMultimodalModel, self).__init__()
        self.bilstm = Bidirectional(GRU(128, return_sequences=False))
        self.fc = Dense(2, activation="softmax")
        self.fc1 = Dense(256)
        self.dropout = Dropout(0.5)

    @tf.function
    def call(self, ts, doc, training=False):
        hiddens = tf.concat([self.bilstm(ts), doc], axis=-1)
        hiddens = self.fc1(hiddens)
        logits = self.fc(self.dropout(hiddens, training=training))
        return (logits, hiddens)


class MultimodalCoAttentionNetwork(tf.keras.models.Model):
    def __init__(self, hidden_dim, co_attention_dim) -> None:
        super(MultimodalCoAttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim

        self.fc1 = Dense(hidden_dim)
        self.fc2 = Dense(hidden_dim)
        self.dropout = Dropout(0.5)

        self.W_b = self.add_weight(
            shape=(self.hidden_dim, self.hidden_dim),
            initializer=initializers.he_normal,
            trainable=True,
            name="mmca_wb",
        )
        self.W_v = self.add_weight(
            shape=(self.co_attention_dim, self.hidden_dim),
            initializer=initializers.he_normal,
            trainable=True,
            name="mmca_wv",
        )
        self.W_q = self.add_weight(
            shape=(self.co_attention_dim, self.hidden_dim),
            initializer=initializers.he_normal,
            trainable=True,
            name="mmca_wq",
        )
        self.W_hv = self.add_weight(
            shape=(self.co_attention_dim, 1),
            initializer=initializers.he_normal,
            trainable=True,
            name="mmca_whv",
        )
        self.W_hq = self.add_weight(
            shape=(self.co_attention_dim, 1),
            initializer=initializers.he_normal,
            trainable=True,
            name="mmca_whq",
        )

    @tf.function
    def call(self, V, Q, training=False):
        """
        :param V: entity lists
        :param Q: time series
        """
        V = tf.transpose(self.fc1(self.dropout(V, training=training)), [0, 2, 1])
        Q = self.fc2(self.dropout(Q, training=training))
        C = tf.matmul(Q, tf.matmul(self.W_b, V))
        H_v = tf.nn.tanh(
            tf.matmul(self.W_v, V)
            + tf.matmul(tf.matmul(self.W_q, tf.transpose(Q, [0, 2, 1])), C)
        )
        H_q = tf.nn.tanh(
            tf.matmul(self.W_q, tf.transpose(Q, [0, 2, 1]))
            + tf.matmul(tf.matmul(self.W_v, V), tf.transpose(C, [0, 2, 1]))
        )
        a_v = tf.nn.softmax(tf.matmul(self.W_hv, H_v, transpose_a=True), axis=-1)
        a_q = tf.nn.softmax(tf.matmul(self.W_hq, H_q, transpose_a=True), axis=-1)
        v = tf.squeeze(tf.matmul(a_v, tf.transpose(V, [0, 2, 1])))
        q = tf.squeeze(tf.matmul(a_q, Q))
        return v, q


class MultimodalGraphModel(tf.keras.models.Model):
    def __init__(self) -> None:
        super(MultimodalGraphModel, self).__init__()
        self.dropout = Dropout(0.5)
        self.fc1 = Dense(256, activation="tanh")
        self.fc2 = Dense(2, activation="softmax")
        self.gcn = GCNConv(256, activation="tanh")
        self.co_attention = MultimodalCoAttentionNetwork(128, 128)
        self.BiGRU = Bidirectional(GRU(128, return_sequences=True))

    @tf.function
    def call(self, ts, doc, entity, index, graph, training=False):
        ts_hiddens = self.BiGRU(ts)
        entity_features, ts_features = self.co_attention(
            entity, ts_hiddens, training=training
        )
        ts_hiddens = ts_hiddens[:, -1, :]
        fusion_hiddens1 = self.fc1(
            self.dropout(
                tf.concat([ts_hiddens, ts_features, doc, entity_features], axis=-1)
            ),
            training=training,
        )
        graph_hiddens = self.gcn([fusion_hiddens1, graph])
        fusion_hiddens2 = tf.concat([ts_hiddens, doc, graph_hiddens], axis=-1)
        logits = self.fc2(
            self.dropout(
                fusion_hiddens2,
                training=training,
            )
        )
        return (logits, fusion_hiddens2)
