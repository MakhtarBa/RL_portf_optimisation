#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. To locate the TensorFlow installation path, do
    the following:

    1. In Python, type 'import tensorflow as tf', then 'print(tf.__file__)'

    2. According to the output, find tensorflow_install_path/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.

    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step
                         to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        self.num_units = num_units
        self.num_proj = num_proj
        self.activation = activation
        self._state_size = (num_units,num_proj)
        self._output_size = num_proj 
        
        w_shape = [num_proj+1,num_units]
        b_shape = [num_units]   
        
        wi = tf.get_variable(name = 'weight_i',shape = w_shape,initializer=tf.glorot_uniform_initializer(seed=1))
        self.wi = wi
        wj = tf.get_variable(name = 'weight_j',shape = w_shape,initializer=tf.glorot_uniform_initializer(seed=1))
        self.wj = wj
        wf = tf.get_variable(name = 'weight_f',shape = w_shape,initializer=tf.glorot_uniform_initializer(seed=1))
        self.wf = wf
        wo = tf.get_variable(name = 'weight_o',shape = w_shape,initializer=tf.glorot_uniform_initializer(seed=1))
        self.wo = wo
        w_project_shape = [num_units,num_proj]
        w = tf.get_variable(name = 'weight_project',shape = w_project_shape,initializer=tf.glorot_uniform_initializer(seed=1))
        
        self.w = w
        
        
        #############################################
        #raise NotImplementedError('Please edit this function.')

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        return self._state_size
        #############################################
        #raise NotImplementedError('Please edit this function.')

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        return self._output_size
        #############################################
        #raise NotImplementedError('Please edit this function.')

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the
        very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
    
    
        #############################################
        #           TODO: YOUR CODE HERE            #
        sigm = math_ops.sigmoid
        tanh = math_ops.tanh
        
        wi = self.wi
        wj = self.wj
        wf = self.wf
        wo = self.wo

        w = self.w
        
        (c_prev, h_prev) = state
        
        i = sigm(math_ops.matmul(array_ops.concat([inputs, h_prev], 1), wi))
        f = sigm(math_ops.matmul(array_ops.concat([inputs, h_prev], 1), wf))
        o = sigm(math_ops.matmul(array_ops.concat([inputs, h_prev], 1), wo))
        j = tanh(math_ops.matmul(array_ops.concat([inputs, h_prev], 1), wj))
        
        c = f * c_prev + i * j
        h_be = o * tanh(c)
        
        h_after = math_ops.matmul(h_be, w) 
        new_state = (c,h_after)
        
        
        return h_after, new_state     


