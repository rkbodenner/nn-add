% Neural network for learning to add pairs of numbers 50 > x >= 0

% Load training examples, created with:
%
% X = floor(rand(1000,2) * 50);
% y = X(:,1) + X(:,2);
% save -text training-examples.mat X y

load('training-examples.mat');

input_layer_size = 2;
hidden_layer_size = 3;
output_layer_size = 1;

% Weights
Theta1 = randInitWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitWeights(hidden_layer_size, output_layer_size);
% Regularization factor
lambda = 0;

J = cost(Theta1, Theta2, input_layer_size, hidden_layer_size, output_layer_size, X, y, lambda);

fprintf('Initial cost: %f\n', J);
