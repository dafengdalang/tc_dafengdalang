import keras
# 优化器与监控值
from keras.optimizers import SGD, RMSprop
from keras.metrics import binary_accuracy, binary_crossentropy
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
# 函数式模型
from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Deconv2D, Dropout
from keras.models import Model  # 函数式模型
from keras.layers.advanced_activations import PReLU
from multi_gpus_util import to_multi_gpu_nodule_segmented, set_gpus
import numpy as np
import pickle
from matplotlib import pyplot as plt
from config import config
import keras.backend as K
import os
import cv2
from dataset import get_img_cubic


def step_decay(epoch, lr):
    if epoch > 0:
        lr = lr * config['network']['lr_decay']
    print("learnrate: ", lr, " epoch: ", epoch)
    return lr

def focal_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return - K.mean(K.pow(y_true_f - y_pred_f, 2) * (y_true_f * K.log(y_pred_f + 1e-5) + (1 - y_true_f) * (K.log(1 - y_pred_f + 1e-5))))

class drawloss(keras.callbacks.Callback):
    def __init__(self, work_dir):
        self.work_dir = work_dir
        self.losses = []
        self.var_losses = []
        if os.path.exists(self.work_dir + 'loss_log.pkl'):
            with open(self.work_dir + 'loss_log.pkl', 'rb') as fp:
                loss_log = pickle.load(fp)
                self.losses = loss_log['losses']
                self.var_losses = loss_log['var_losses']

    # def on_train_begin(self, logs=None):
    #     pass
    #
    # def on_train_end(self, logs=None):
    #     pass

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.var_losses.append(logs['val_loss'])
        with open(self.work_dir + 'loss_log.pkl', 'wb') as fp:
            loss_log = {'losses': self.losses, 'var_losses': self.var_losses}
            pickle.dump(loss_log, fp)

        plt.plot(self.losses)
        plt.plot(self.var_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.work_dir + "train_loss.png")
        plt.close()


class CNNModel(object):
    def __init__(self, load_weight_path=None, train=False, learning_rate=0.01, gpu_nums=3):
        self.train = train
        self.gpu_nums = gpu_nums

        # GPU 版本
        set_gpus(num_gpus=self.gpu_nums, auto_growth=True)
        
        self.model = self.build_model(load_weight_path)
        if self.train:
            self.model.compile(optimizer = RMSprop(lr=learning_rate),
                               loss = {"out_class": focal_loss},
                               metrics = {'out_class' : [focal_loss, binary_accuracy]})
            # self.model.compile(optimizer=SGD(lr = learning_rate, momentum = 0.9, nesterov = True), loss={"out_class": "binary_crossentropy"}, metrics={"out_class": [binary_accuracy, binary_crossentropy]})

    def build_model(self, load_weight_path=None) -> Model:
        self.inputs = Input(config['network']['input_shape'], name="inputs")
        self.outputs = self.network(self.inputs)
        model = Model(inputs=self.inputs, outputs=self.outputs)
        print(model.summary())
        
        # 多GPU分解
        model = to_multi_gpu_nodule_segmented(model, n_gpus=self.gpu_nums)

        if load_weight_path is not None:
            print("load fpr model from: ", load_weight_path)
            print("start loading fpr model...")
            model.load_weights(load_weight_path, by_name=False)
            print("end loading fpr model")

        return model

    def network(self, inputs):
        # 256 256 3
        l = Conv2D(64, (3, 3), padding = 'same', name = 'conv1_1')(inputs)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(64, (3, 3), padding = 'same', name = 'conv1_2')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(64, (2, 2), strides = (2, 2), padding = 'valid', name = 'down_sample_1')(l)
        # 128 128 64

        l = Conv2D(128, (3, 3), padding = 'same', name = 'conv2_1')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(128, (3, 3), padding = 'same', name = 'conv2_2')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(128, (2, 2), strides = (2, 2), padding = 'valid', name = 'down_sample_2')(l)
        # 64 64 128

        l = Conv2D(256, (3, 3), padding = 'same', name = 'conv3_1')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(256, (3, 3), padding = 'same', name = 'conv3_2')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(256, (2, 2), strides = (2, 2), padding = 'valid', name = 'down_sample_3')(l)
        # 32 32 256

        l = Conv2D(512, (3, 3), padding = 'same', name = 'conv4_1')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(512, (3, 3), padding = 'same', name = 'conv4_2')(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Conv2D(512, (2, 2), strides = (2, 2), padding = 'valid', name = 'down_sample_4')(l)
        # 16 16 512

        l = Flatten()(l)

        l = Dense(200)(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)
        l = Dropout(0.2)(l)

        l = Dense(100)(l)
        l = BatchNormalization()(l)
        l = PReLU()(l)

        l = Dense(2, activation = 'sigmoid', name = 'out_class')(l)
        
        return l

    def train_model(self, train_data_gen, train_data_num, eval_data_gen, eval_data_num, batch_size, train_dir=None, model_name='cnn', initial_epoch=0):
        if self.train == False:
            print('this model is not in training mode!!!')
            return
        print('train data num: %d' % train_data_num)
        print('eval data num: %d' % eval_data_num)
        if batch_size < self.gpu_nums:
            batch_size = self.gpu_nums
        # monitor：需要监视的值；verbose：信息展示模式，0或1；save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
        checkpoint = ModelCheckpoint(
            train_dir + model_name + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5",
            monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
            period=1)  # 临时模型文件

        checkpoint_fixed_name = ModelCheckpoint(train_dir + model_name + "_best.hd5",
                                                monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=False, mode='auto', period=1)
        learnrate_scheduler = LearningRateScheduler(step_decay)

        draw_loss = drawloss(train_dir)
        self.model.fit_generator(generator=train_data_gen, steps_per_epoch=train_data_num // batch_size, epochs=20,
                                 validation_data=eval_data_gen,
                                 validation_steps=(eval_data_num + batch_size - 1) // batch_size,
                                 callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler, draw_loss],
                                 initial_epoch=initial_epoch)
        self.model.save(train_dir + model_name + "_end.hd5")

    def predict(self, inputs, batch_size):
        inputs_num = len(inputs)

        if batch_size < self.gpu_nums:
            batch_size = self.gpu_nums

        if inputs_num < self.gpu_nums:
            inputs_shape = inputs.shape
            add_zeros = np.zeros([self.gpu_nums - inputs_shape[0]] + list(inputs_shape[1:]))
            inputs = np.concatenate([inputs, add_zeros], 0)

        preds = self.model.predict(inputs, batch_size=batch_size)
        if inputs_num < self.gpu_nums:
            preds = preds[: inputs_num, :, :, :]

        return preds
    def predict_one_img(self, image_path, batch_size):
        image_array = cv2.imread(image_path)
        size_y, size_x, channel = image_array.shape
        point_list = [(x, y) for x in range(0, size_x, 128) for y in range(0, size_y, 128)]
        data = []
        pred_list = []
        for point in point_list:
            data.append(np.array(get_img_cubic(image_array, point) / 255, np.float32))
        
        point_num = len(point)

        for s in range(0, point_num, batch_size):
            batch_data = data[s : min(point_num, s + batch_size)]
            batch_pred = self.predict(np.array(batch_data, np.float32), batch_size)
            for pred in batch_pred:
                pred_list.append(pred[1])
        
        return max(pred_list)

if __name__ == '__main__':
    # test model frame
    fpr_models = CNNModel(None, False)
    inputs = np.ones([16, 256, 256, 3], dtype=np.float32)
    preds = fpr_models.predict(inputs, batch_size=16)
    print(preds)
