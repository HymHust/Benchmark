import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model, initializers
from tensorflow.keras import layers
from keras.layers import Conv1D,  Dropout
from keras.regularizers import l2

# 类别数
num_classes = 3
signal_size = [5120,1]
patch_size = [512,1]
conv_size = 512
rate=2
stride=int(conv_size/rate)
num_patches = int((signal_size[0]-patch_size[0])/stride)+1
projection_dim = 64
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]
k=10
weight_layers=1
transformer_layers = 5



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x



class ConcatClassTokenAddPosEmbed(layers.Layer):
    def __init__(self, embed_dim=40, num_patches=40, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def build(self, input_shape):
        self.cls_token = self.add_weight(name="cls",
                                         shape=[1,1, self.embed_dim],
                                         initializer=initializers.Zeros(),
                                         trainable=True,
                                         dtype=tf.float32)
        self.pos_embed = self.add_weight(name="pos_embed",
                                         shape=[1,self.num_patches + 1, self.embed_dim],
                                         initializer=initializers.RandomNormal(stddev=0.02),
                                         trainable=True,
                                         dtype=tf.float32)

    def call(self, inputs, **kwargs):
        batch_size= tf.shape(inputs)[0]

        # [1, 1, 768] -> [B, 1, 768]
        cls_token = tf.broadcast_to(self.cls_token, shape=[batch_size, 1, self.embed_dim])
        x = tf.concat([cls_token, inputs], axis=1)  # [B, 197, 768]
        x = x + self.pos_embed

        return x
    def get_config(self):  #在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"embed_dim": self.embed_dim,"num_patches":self.num_patches}
        base_config = super(ConcatClassTokenAddPosEmbed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def MPI(inputs, filters=projection_dim, kernerl_size=3, strides=1, conv_padding='same'):
    x = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides, dilation_rate=1,
               padding=conv_padding, kernel_regularizer=l2(1e-6), activation='relu')(inputs)
    y = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides, dilation_rate=2,
               padding=conv_padding, kernel_regularizer=l2(1e-6), activation='relu')(inputs)
    z = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides, dilation_rate=3,
               padding=conv_padding, kernel_regularizer=l2(1e-6), activation='relu')(inputs)

    combined = tf.keras.layers.concatenate([x[:, :, :, tf.newaxis], y[:, :, :, tf.newaxis], z[:, :, :, tf.newaxis]],
                                           axis=-1)
    combined = tf.reduce_mean(combined, axis=-1)
    return combined
class PatchEncoderCUM(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        super(PatchEncoderCUM, self).__init__()
        self.num_patches = num_patches
        self.cum = Conv1D(filters=projection_dim, kernel_size=conv_size, strides=stride,padding='valid',dilation_rate=1,
                           kernel_regularizer=l2(1e-6), activation=tf.nn.gelu)
        #self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    # 这里call后需要定义get_config函数，命名自拟，文章3.9中给出

    def call(self, patch, **kwargs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        #encoded = [self.cum1(patch),self.cum2(patch),self.cum3(patch)] + self.position_embedding(positions)*3
        encoded = self.cum(patch) + self.position_embedding(positions)
        return encoded
    def get_config(self):  #在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"num_patches": self.num_patches,"cum":self.cum,'position_embedding':self.position_embedding}
        base_config = super(PatchEncoderCUM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def CTL(inputs,k):
    crop = ConcatClassTokenAddPosEmbed(embed_dim=projection_dim, num_patches=num_patches)(inputs)
    #crop=encoded_patches
    temporary = crop
    for _ in range(weight_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(crop)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, crop])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        crop = layers.Activation('tanh')(x3)
    weights=tf.reduce_sum(crop,axis=-1)
    top_values, top_indices = tf.math.top_k(weights[-1], k,name='topk')#这里需要排序，默认是True，不排序效果略差
    selected_values = tf.gather(temporary, top_indices,axis=1,name='selected_patches')
    return selected_values,crop


def CFSPT(input_shape):
    inputs = layers.Input(shape=input_shape)
    add_pos_embed=PatchEncoderCUM(num_patches, projection_dim)(inputs)
    add_pos_embed = MPI(add_pos_embed, filters=projection_dim, kernerl_size=3, strides=1, conv_padding='same')
    add_pos_embed,ALL=CTL(add_pos_embed,k)
    # 创建多个Transformer encoding 块
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(add_pos_embed)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection.
        x2 = layers.Add()([attention_output, add_pos_embed])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.01)
        add_pos_embed = layers.Add()([x3, x2])

    features = Dropout(0.1)(add_pos_embed)
    features = layers.GlobalAveragePooling1D()(features)

    logits = layers.Dense(units=num_classes, activation='softmax')(features)
    # 构建
    model = Model(inputs=inputs, outputs=logits)
    model.summary()
    plot_model(model, to_file='CFSPT.png', show_shapes=True, show_layer_names='False', rankdir='TB')
    return model


if __name__ == "__main__":
    Model= CFSPT([5120,1])

