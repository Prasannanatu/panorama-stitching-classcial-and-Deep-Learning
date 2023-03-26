import tensorflow as tf
from tensorflow.keras import models

# Load the model
model = models.load_model('/home/pvnatu/venv/bin/~venv/YourDirectoryID_p1/Phase2/Checkpoints/29a0unsup2model.ckpt')

# Get the TensorFlow default graph
graph = tf.get_default_graph()

# Get the final output node in the graph
output_node = [n.name for n in graph.as_graph_def().node if n.op == 'Softmax'][0]

# Visualize the model using make_dot
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96, subgraph=False)
