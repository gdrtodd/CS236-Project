# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module defines the softranks and softsort operators."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v2 as tf
from soft_sort import soft_quantilizer


DIRECTIONS = ('ASCENDING', 'DESCENDING')


def _preprocess(x, axis):
  """Reshapes the input data to make it rank 2 as required by SoftQuantilizer.

  The SoftQuantilizer expects an input tensor of rank 2, where the first
  dimension is the batch dimension and the soft sorting is applied on the second
  one.

  Args:
   x: Tensor<float> of any dimension.
   axis: (int) the axis to be turned into the second dimension.

  Returns:
   a Tensor<float>[batch, n] where n is the dimensions over the axis and batch
   the product of all other dimensions
  """
  dims = list(range(x.shape.rank))
  dims[-1], dims[axis] = dims[axis], dims[-1]
  z = tf.transpose(x, dims) if dims[axis] != dims[-1] else x
  return tf.reshape(z, (-1, tf.shape(x)[axis]))


def _postprocess(x, shape, axis):
  """Applies the inverse transformation of _preprocess.

  Args:
   x: Tensor<float>[batch, n]
   shape: TensorShape of the desired output.
   axis: (int) the axis along which the original tensor was processed.

  Returns:
   A Tensor<float> with the shape given in argument.
  """
  s = list(shape)
  s[axis], s[-1] = s[-1], s[axis]
  z = tf.reshape(x, s)

  # Transpose to get back to the original shape
  dims = list(range(shape.rank))
  dims[-1], dims[axis] = dims[axis], dims[-1]
  return tf.transpose(z, dims) if dims[axis] != dims[-1] else z


def softsort(x, direction='ASCENDING', axis=-1, **kwargs):
  """Applies the softsort operator on input tensor x.

  This operator acts as differentiable alternative to tf.sort.

  Args:
   x: the input tensor. It can be either of shape [batch, n] or [n].
   direction: the direction 'ASCENDING' or 'DESCENDING'
   axis: the axis on which to operate the sort.
   **kwargs: see SoftQuantilizer for possible parameters.

  Returns:
   A tensor of the same shape as the input.
  """
  if direction not in DIRECTIONS:
    raise ValueError('`direction` should be one of {}'.format(DIRECTIONS))

  z = _preprocess(x, axis)
  descending = (direction == 'DESCENDING')
  sorter = soft_quantilizer.SoftQuantilizer(z, descending=descending, **kwargs)

  # In case we are applying some quantization while sorting, the number of
  # outputs should be the number of targets.
  shape = x.shape.as_list()
  shape[axis] = sorter.target_weights.shape[1]
  return _postprocess(sorter.softsort, tf.TensorShape(shape), axis)


def softranks(x, direction='ASCENDING', axis=-1, zero_based=True, **kwargs):
  """A differentiable argsort-like operator that returns directly the ranks.

  Note that it behaves as the 'inverse' of the argsort operator since it returns
  soft ranks, i.e. real numbers that play the role of indices and quantify the
  relative standing (among all n entries) of each entry of x.

  Args:
   x: Tensor<float> of any shape.
   direction: (str) either 'ASCENDING' or 'DESCENDING', as in tf.sort.
   axis: (int) the axis along which to sort, as in tf.sort.
   zero_based: (bool) to return values in [0, n-1] or in [1, n].
   **kwargs: see SoftQuantilizer for possible parameters.

  Returns:
   A Tensor<float> of the same shape as the input containing the soft ranks.
  """
  if direction not in DIRECTIONS:
    raise ValueError('`direction` should be one of {}'.format(DIRECTIONS))

  descending = (direction == 'DESCENDING')
  z = _preprocess(x, axis)
  sorter = soft_quantilizer.SoftQuantilizer(z, descending=descending, **kwargs)
  ranks = sorter.softcdf * tf.cast(tf.shape(z)[1], dtype=x.dtype)
  if zero_based:
    ranks -= tf.cast(1.0, dtype=x.dtype)

  return _postprocess(ranks, x.shape, axis)


def softquantiles(x, quantiles, quantile_width=None, axis=-1, **kwargs):
  """Computes a (single) soft quantile via optimal transport.

  This operator takes advantage of the fact that an exhaustive softsort is not
  required to recover a single quantile. Instead, one can transport all
  input values in x onto only 3 weighted values. Target weights are adjusted so
  that those values in x that are transported to the middle value in the target
  vector y correspond to those concentrating around the quantile of interest.

  This idea generalizes to more quantiles, interleaving small weights on the
  quantile indices and bigger weights in between, corresponding to the gap from
  one desired quantile to the next one.

  Args:
   x: Tensor<float> of any shape.
   quantiles: list<float> the quantiles to be returned. It can also be a single
    float.
   quantile_width: (float) mass given to the bucket supposed to attract points
    whose value concentrate around the desired quantile value. Bigger width
    means that we allow the soft quantile to be a mixture of
    more points further away from the quantile. If None, the width is set at 1/n
    where n is the number of values considered (the size along the 'axis').
   axis: (int) the axis along which to compute the quantile.
   **kwargs: see SoftQuantilizer for possible extra parameters.

  Returns:
    A Tensor<float> similar to the input tensor, but the axis dimension is
    replaced by the number of quantiles specified in the quantiles list.
    Hence, if only a quantile is requested (quantiles is a float) only one value
    in that axis is returned. When several quantiles are requested, the tensor
    will have that many values in that axis.

  Raises:
    tf.errors.InvalidArgumentError when the quantiles and quantile width are not
    correct, namely quantiles are either not in sorted order or the
    quantile_width is too large.

  """
  if isinstance(quantiles, float):
    quantiles = [quantiles]
  quantiles = tf.constant(quantiles, tf.float32)

  # Preprocesses submitted quantiles to check that they satisfy elementary
  # constraints.
  valid_quantiles = tf.boolean_mask(
      quantiles, tf.logical_and(quantiles > 0.0, quantiles < 1.0))
  num_quantiles = tf.shape(valid_quantiles)[0]

  # Includes values on both ends of [0,1].
  extended_quantiles = tf.concat([[0.0], valid_quantiles, [1.0]], axis=0)

  # Builds filler_weights in between the target quantiles.
  filler_weights = extended_quantiles[1:] - extended_quantiles[:-1]
  if quantile_width is None:
    quantile_width = tf.reduce_min(
        tf.concat(
            [filler_weights, [1.0 / tf.cast(tf.shape(x)[axis], dtype=x.dtype)]],
            axis=0))

  # Takes into account quantile_width in the definition of weights
  shift = -tf.ones(tf.shape(filler_weights), dtype=x.dtype)
  shift = shift + 0.5 * (
      tf.one_hot(0, num_quantiles + 1) +
      tf.one_hot(num_quantiles, num_quantiles + 1))
  filler_weights = filler_weights + quantile_width * shift

  assert_op = tf.Assert(tf.reduce_all(filler_weights >= 0.0), [filler_weights])
  with tf.control_dependencies([assert_op]):
    # Adds one more value to have tensors of the same shape to interleave them.
    quantile_weights = tf.ones(num_quantiles + 1) * quantile_width

    # Interleaves the filler_weights with the quantile weights.
    weights = tf.reshape(
        tf.stack([filler_weights, quantile_weights], axis=1), (-1,))[:-1]

    # Sends only the positive weights to the softsort operator.
    positive_weights = tf.boolean_mask(weights, weights > 0.0)
    result = softsort(
        x,
        direction='ASCENDING', axis=axis, target_weights=positive_weights,
        **kwargs)

    # Recovers the indices corresponding to the desired quantiles.
    odds = tf.math.floormod(tf.range(weights.shape[0], dtype=tf.float32), 2)
    positives = tf.cast(weights > 0.0, tf.float32)
    indices = tf.cast(tf.math.cumsum(positives) * odds, dtype=tf.int32)
    indices = tf.boolean_mask(indices, indices > 0) - 1
    result = tf.gather(result, indices, axis=axis)

    # In the specific case where we want a single quantile, squeezes the
    # quantile dimension.
    return tf.cond(tf.equal(tf.shape(result)[axis], 1),
                   lambda: tf.squeeze(result, axis=axis),
                   lambda: result)
