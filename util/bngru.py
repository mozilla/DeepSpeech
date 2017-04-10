import tensorflow as tf
import numpy as np

from tensorflow.python.util import nest

RNNCell = tf.contrib.rnn.RNNCell

class BNGRUCell(RNNCell):
    """Batch normalized, Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, is_training_tensor, max_bn_steps, initial_scale=0.1, decay=0.9, activation=tf.tanh):
        self._num_units = num_units
        self._training = is_training_tensor
        self._max_bn_steps = max_bn_steps
        self._initial_scale = initial_scale
        self._decay = decay
        self._activation = activation

    @property
    def state_size(self):
        return (self._num_units, 1)

    @property
    def output_size(self):
        return self._num_units

    def _batch_norm(self, x, name_scope, step, epsilon=1e-5, no_offset=False, set_forget_gate_bias=False):
        with tf.variable_scope(name_scope):
            size = x.get_shape().as_list()[1]

            scale = tf.get_variable('scale', [size], initializer=tf.constant_initializer(self._initial_scale))
            if no_offset:
                offset = 0
            elif set_forget_gate_bias:
                offset = tf.get_variable('offset', [size], initializer=offset_initializer())
            else:
                offset = tf.get_variable('offset', [size], initializer=tf.zeros_initializer())

            pop_mean_all_steps = tf.get_variable('pop_mean', [self._max_bn_steps, size], initializer=tf.zeros_initializer(), trainable=False)
            pop_var_all_steps = tf.get_variable('pop_var', [self._max_bn_steps, size], initializer=tf.ones_initializer(), trainable=False)

            step = tf.minimum(step, self._max_bn_steps - 1)

            pop_mean = pop_mean_all_steps[step]
            pop_var = pop_var_all_steps[step]

            batch_mean, batch_var = tf.nn.moments(x, [0])

            def batch_statistics():
                pop_mean_new = pop_mean * self._decay + batch_mean * (1 - self._decay)
                pop_var_new = pop_var * self._decay + batch_var * (1 - self._decay)
                with tf.control_dependencies([pop_mean.assign(pop_mean_new), pop_var.assign(pop_var_new)]):
                    return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

            def population_statistics():
                return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

            return tf.cond(self._training, batch_statistics, population_statistics)

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with self._num_units cells."""
        with tf.variable_scope(scope or "gru_cell"):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                h, step = state
                _step = tf.squeeze(tf.gather(tf.cast(step, tf.int32), 0))
                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(
                    value=_linear(
                        [inputs, h], 2 * self._num_units, bias=False, scope=scope),
                    num_or_size_splits=2,
                    axis=1)

                r = self._batch_norm(r, 'r', _step, no_offset=True)

                r, u = tf.sigmoid(r), tf.sigmoid(u)
            with tf.variable_scope("candidate"):
                c = self._activation(_linear([inputs, r * h],
                                             self._num_units,
                                             bias=False,
                                             scope=scope))
            new_h = u * h + (1 - u) * c
        return new_h, (new_h, step+1)


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = tf.get_variable_scope()
  with tf.variable_scope(scope) as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)