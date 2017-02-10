import tensorflow as tf


class TFLogger(object):
    """ Creates an "empty model" that writes Tensorflow summaries. Can
        visualize these summaries with Tensorboard.
    """
    def __init__(self, summary_dir, pytorch=False):
        super(TFLogger, self).__init__()
        self.summary_dir = summary_dir
        self.pytorch = pytorch
        if pytorch: import torch
        self.__initialize()

    def __initialize(self):
        sess = tf.Session()
        loss = tf.Variable(0.0, name="loss", trainable=False)
        acc = tf.Variable(0.0, name="accuracy", trainable=False)
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", acc)
        summary_op = tf.summary.merge([loss_summary, acc_summary])
        summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        self.sess = sess
        self.summary_op = summary_op
        self.summary_writer = summary_writer
        self.loss = loss
        self.acc = acc

    def log(self, step, loss, accuracy):
        if self.pytorch:
            loss = loss.data.numpy()
            accuracy = accuracy.data.numpy()
        feed_dict = {
            self.loss: loss,
            self.acc: accuracy,
        }

        # sess.run returns a list, so we have to explicitly
        # extract the first item using sess.run(...)[0
        #import ipdb; ipdb.set_trace()
        summaries = self.sess.run([self.summary_op], feed_dict)[0]
        self.summary_writer.add_summary(summaries, step)
