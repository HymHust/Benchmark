import preprocess
import Model
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
import numpy as np
from keras.optimizers import adam_v2
import tensorflow as tf


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
sess=tf.compat.v1.Session(config=config)

batch_size =64
epochs = 20
num_classes = 3
length = 5120
BatchNorm = True # 是否批量归一化
number = 200# 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

def data_pre(path,number):
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path, length=length,
                                                                           number=number,
                                                                           normal=normal,
                                                                           rate=rate,
                                                                           enc=False, enc_step=28)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')
    x_valid = x_valid.astype('float32')
    y_valid = y_valid.astype('float32')
    x_test = x_test.astype('float32')
    y_test = y_test.astype('float32')
    x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :,np.newaxis], x_test[:, :, np.newaxis]
    #x_train, x_valid, x_test = x_train[:, :, np.newaxis, np.newaxis], x_valid[:, :, np.newaxis, np.newaxis], x_test[:, :, np.newaxis, np.newaxis]
    return x_train,x_valid,x_test,y_train,y_valid,y_test

if __name__ == '__main__':
    path = r".\Train"
    path_test = r".\Test"
    x_train,x_valid,x_test,y_train,y_valid,y_test = data_pre(path,200)
    x_train1,x_valid1,x_test1,y_train1,y_valid1,y_test1 = data_pre(path_test,200)
    # 输入数据的维度
    input_shape =x_train.shape[1:]
    print('训练样本维度:', x_train.shape)
    print(x_train.shape[0], '训练样本个数')
    print('验证样本的维度', x_valid.shape)
    print(x_valid.shape[0], '验证样本个数')
    print('测试样本的维度', x_test.shape)
    print(x_test.shape[0], '测试样本个数')

    model = Model.CFSPT(input_shape)
    #model.summary()
    # 定义优化器
    #Nadam1 = Nadam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,schedule_decay=0.004)
    adam = adam_v2.Adam(lr=0.001)
    # 定义优化器，loss function, 训练过程中计算准确率
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
    # 画出网络结构
    # plot_model(model, to_file='model_cnn.png', show_shapes=True, show_layer_names='False', rankdir='TB')
    callback_list = [ModelCheckpoint(filepath='CFSPT.hdf5', verbose=1, save_best_only=True,monitor='val_acc',mode='auto')]
    # 训练模型
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
              callbacks=callback_list)

    model.load_weights('CFSPT.hdf5')
    # 评估模型
    #model.summary()
    loss, acc = model.evaluate(x_train1, y_train1)
    print("loss:", loss)
    print("accuracy", acc)





