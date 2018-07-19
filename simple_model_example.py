import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class simpleModel(object):
    def __init__(self, weight_path = None, train = False):
        # with tf.Session() as self.sess:
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape = [None, 1], name = 'input_point')
        
        self.layer1_w = tf.get_variable('l1_w', shape = [1, 100], dtype = tf.float32, initializer = tf.random_normal_initializer(0.0, 0.01))
        self.layer1_b = tf.get_variable('l1_b', shape = [100], dtype = tf.float32, initializer = tf.random_normal_initializer(0.0, 0.01))
        self.layer1 = tf.nn.leaky_relu(tf.matmul(self.input, self.layer1_w) + self.layer1_b, name = 'l1')

        self.layer2_w = tf.get_variable('l2_w', shape = [100, 100], dtype = tf.float32, initializer = tf.random_normal_initializer(0.0, 0.01))
        self.layer2_b = tf.get_variable('l2_b', shape = [100], dtype = tf.float32, initializer = tf.random_normal_initializer(0.0, 0.01))
        self.layer2 = tf.nn.leaky_relu(tf.matmul(self.layer1, self.layer2_w) + self.layer2_b, name = 'l2')

        self.layer3_w = tf.get_variable('l3_w', shape = [100, 1], dtype = tf.float32, initializer = tf.random_normal_initializer(0.0, 0.01))
        self.layer3_b = tf.get_variable('l3_b', shape = [1], dtype = tf.float32, initializer = tf.random_normal_initializer(0.0, 0.01))
        self.layer3 = tf.nn.leaky_relu(tf.matmul(self.layer2, self.layer3_w) + self.layer3_b, name = 'l3')

        self.output = self.layer3
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        if weight_path:
            self.saver.restore(self.sess, weight_path)
        else:
            self.sess.run(tf.global_variables_initializer())

        if train:
            self.input_y = tf.placeholder(tf.float32, shape = [None, 1], name = 'input_value')
            self.loss = tf.nn.l2_loss(self.input_y - self.output)
            self.loss_vision = tf.summary.scalar('loss', self.loss)
            self.opt = tf.train.GradientDescentOptimizer(0.001).minimize(self.loss)
        self.merged_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('D:\\data\\abc_board', self.sess.graph)
    
    def trainModel(self):
        for epoch in range(150):
            loss = 0
            summary = None
            for batch in range(0, 1000):
                x = np.random.random(size = [100, 1])
                # print(x)
                y = np.array(np.sin(x * np.pi * 2)) + (np.random.random(size = [100, 1]) - 0.5) * 0.1
                _, loss, summary = self.sess.run(fetches = [self.opt, self.loss, self.merged_op], feed_dict = {self.input: x, self.input_y: y})
                if batch % 100 == 0:
                    print('epoch%d, batch%d, loss: %f' % (epoch, batch, loss))
            self.saver.save(self.sess, 'D:\\data\\abc\\epoch_%d.ckpt' % epoch)
            self.summary_writer.add_summary(summary, epoch)

    def predict(self, x):
        output = self.sess.run(self.output,{self.input: x})
        return output

if __name__ == '__main__':
    if True:
        model = simpleModel(train = True)
        model.trainModel()
    if False:
        model = simpleModel('D:\\data\\abc\\epoch_149.ckpt',train = False)
        x = np.array([[z / 1000.0] for z in range(1000)])
        y = model.predict(x)
        xx = []
        yy = []
        # print(y)
        for x_i in x:
            xx.append(x_i[0])
        for y_i in y:
            yy.append(y_i[0])
        # print(yy)
        # print('aaaaa')
        plt.plot(xx,yy)
        plt.show()
