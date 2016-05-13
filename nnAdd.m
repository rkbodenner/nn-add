% Neural network for learning to add pairs of numbers 50 > x >= 0

% Load training examples, created with:
%
% X = floor(rand(1000,2) * 50);
% y = X(:,1) + X(:,2);
% save -text training-examples.mat X y

load('training-examples.mat');

input_layer_size = 2;
hidden_layer_size = 3;
output_layer_size = 100;

% Weights. These seem good.
Theta1 = [
  0.101237  -0.045077 -0.069981 ;
  0.030126 -0.071757 -0.116876 ;
  0.004693 -0.044443 0.007245
];

Theta2 = [
  0.032645 -0.061070 0.064763 -0.066527
];

% or randomly...
Theta1 = randInitWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitWeights(hidden_layer_size, output_layer_size);

fprintf("Initial weights\n");
fprintf("Theta1\n");
fprintf("%f\n", Theta1);
fprintf("\nTheta2\n");
fprintf("%f\n", Theta2);
fprintf("\n");

% Regularization factor
lambda = 0;

init_params = [Theta1(:) ; Theta2(:)];
J = cost(init_params, input_layer_size, hidden_layer_size, output_layer_size, X, y, lambda);

fprintf('Initial cost: %f\n', J);

% Training
optopts = optimset('MaxIter', 50);
costFn = @(p) cost(p, ...
  input_layer_size, hidden_layer_size, output_layer_size, ...
  X, y, lambda);

[params, J] = fmincg(costFn, init_params, optopts);
fprintf('Training complete. Final cost: %f\n', J);

[final_Theta1, final_Theta2] = paramMatrixify(params, input_layer_size, hidden_layer_size, output_layer_size);
h = predict(final_Theta1, final_Theta2, X);
