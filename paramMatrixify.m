function [Theta1, Theta2] = paramMatrixify(param_vector, ...
  input_layer_size, hidden_layer_size, output_layer_size)
%PARAMMATRIXIFY returns parameters to a two-layer neural network as a pair of matrices, given
%an unrolled vector of the params and the architecture of the network.

Theta1 = reshape(
  param_vector(1:hidden_layer_size * (input_layer_size + 1)), ...
  hidden_layer_size, (input_layer_size + 1)
);
Theta2 = reshape(
  param_vector((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
  output_layer_size, (hidden_layer_size + 1)
);
