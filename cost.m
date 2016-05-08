function [J, grad] = cost(...
  params, ...
  input_layer_size, hidden_layer_size, output_layer_size, ...
  X, Y, lambda)
%COST implements the cost function and gradient computation for a two-layer
%neural network. It returns the cost and the gradients suitable for use with
%Matlab optimization solvers.

% Number of training examples
m = size(X, 1);

[Theta1, Theta2] = paramMatrixify(params, input_layer_size, hidden_layer_size, output_layer_size);

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

% Backpropagation
for t = 1:m
  delta_L3 = a3(:,t) - Y(t,:);
  delta_L2 = (Theta2(:,2:end)' * delta_L3) .* sigmoidGradient(z2(:,t));
  Theta2_grad = Theta2_grad + delta_L3 * a2(:,t)';
  Theta1_grad = Theta1_grad + delta_L2 * a1(t,:);
endfor

Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
