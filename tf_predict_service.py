import tensorflow as tf


class PredictService(object):

    GRAPH_PB_PATH = './model'

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=tf.Graph(), config=config)
        tf.saved_model.loader.load(
            self.session,
            [tf.saved_model.tag_constants.SERVING],
            self.GRAPH_PB_PATH)
        self.write_tensorboard(self.session)
        # self.print_layer(self.session.graph)
        self.x = self.session.graph.get_tensor_by_name('Placeholder:0')
        self.y = self.session.graph.get_tensor_by_name('scores:0')
        label_tensor = self.session.graph.get_tensor_by_name('Const_1:0')
        self.label = self.session.run(label_tensor)

    def predict(self, img: []):
        pred = self.session.run(self.y, feed_dict={
            self.x: img,
        })
        return self.label, pred

    @staticmethod
    def write_tensorboard(sess: tf.Session):
        writer = tf.summary.FileWriter('./log/')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()

    @staticmethod
    def print_layer(graph):
        for op in graph.get_operations():
            print(op.name)
