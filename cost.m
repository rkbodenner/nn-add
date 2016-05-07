function [J, grad] = cost(...
  Theta1, Theta2, ...
  input_layer_size, hidden_layer_size, output_layer_size, ...
  X, Y, lambda)
%COST implements the cost function for a two-layer neural network

% Number of training examples
m = size(X, 1);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward
a1 = [ones(m, 1) X];
z2 = Theta1 * a1';
a2 = [ones(1, m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
H = a3;

% Cost
for i = 1:m
  for k = 1:output_layer_size
    J = J + ( (-Y(i,k) * log(H(k,i))) - ((1-Y(i,k)) * log(1-H(k,i))));
  endfor
endfor
J = (1/m) * J;

grad = [];

end
