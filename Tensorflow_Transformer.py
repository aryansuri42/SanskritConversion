import numpy as np
import tensorflow as tf

def positional_encoding(length, depth):
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis,:]/depth

    angle_rates = 1/(10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_model, embedding=None):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) if embedding is None else embedding
        self.pos_encoding = positional_encoding(length = 2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):

        return self.embedding.compute_mask(*args,**kwargs)
    
    def call(self, x):
        x=self.embedding(x)
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(query=x, key=context, value=context, return_attention_scores=True)
        self.attn_scores = attn_scores
        x=self.add([x,attn_output])
        x=self.layernorm(x)

        return x
    
class GlobalSelfAttention(BaseAttention):

    def call(self, x):

        attn_output = self.mha(query=x, value=x, key=x)
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return x
    

class CausalSelfAttention(BaseAttention):

    def call(self, x):

        attn_output = self.mha(query=x, key=x, value=x, use_causal_mask=True)
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation="relu"),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.self_attention = GlobalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x=self.self_attention(x)
        x=self.ffn(x)
        return x
    

class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):

        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layer = [
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        })
        return config

    def call(self, x):
        x=self.pos_embedding(x)
        x=self.dropout(x)
        for i in range(self.num_layers):
            x=self.enc_layer[i](x)

        return x
    
class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads = num_heads,
            key_dim = d_model,
            dropout = dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x=self.causal_self_attention(x=x)
        x=self.cross_attention(x=x, context=context)

        self.last_attn_scores = self.cross_attention.attn_scores
        x=self.ffn(x)
        return x
    
class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.d_model=d_model
        self.num_layers=num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate
            )for _ in range(num_layers)
        ]
        self.last_attn_scores=None

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        })
        return config

    def call(self, x, context):
        x=self.pos_embedding(x)
        x=self.dropout(x)

        for i in range(self.num_layers):
            x=self.dec_layers[i](x,context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x
    
def Transformer(
        input_vocab_size,
        target_vocab_size,
        encoder_input_size,
        decoder_input_size,
        num_layers,
        d_model,
        num_heads,
        dff,
        dropout_rate=0.1
):
    inputs = [
        tf.keras.layers.Input(shape=(encoder_input_size,), dtype=tf.int64),
        tf.keras.layers.Input(shape=(decoder_input_size,), dtype=tf.int64)
    ]
    encoder_input, decoder_input = inputs
    encoder=Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=input_vocab_size, dropout_rate=dropout_rate)(encoder_input)
    decoder=Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff, vocab_size=target_vocab_size, dropout_rate=dropout_rate)(decoder_input, encoder)
    output = tf.keras.layers.Dense(target_vocab_size)(decoder)
    return tf.keras.Model(inputs=inputs, outputs=output)


