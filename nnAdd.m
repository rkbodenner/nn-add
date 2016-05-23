% Neural network for learning to add pairs of positive numbers within a limited interval

max_operand_value = 1;

% Load training examples, created with:
%
X = floor(rand(100,2) * (max_operand_value+1));
y = X(:,1) + X(:,2);
%save -text training-examples.mat X y
%load('training-examples.mat');

m = size(X,1);
mtrain = 0.6 * m;
mval = 0.2 * m;
mtest = m - mtrain - mval;
Xtrain = X(1:mtrain,:);
Xval = X(mtrain+1:mtrain+mval,:);
Xtest = X(mtrain+mval+1:m,:);
ytrain = y(1:mtrain,:);
yval = y(mtrain+1:mtrain+mval,:);
ytest = y(mtrain+mval+1:m,:);

input_layer_size = 2;
hidden_layer_size = 3;
output_layer_size = (max_operand_value*2)+1;  % One label per possible output value

fprintf("Architecture: %d -> %d -> %d\n", input_layer_size, hidden_layer_size, output_layer_size);

Theta1 = randInitWeights(input_layer_size, hidden_layer_size);
Theta2 = randInitWeights(hidden_layer_size, output_layer_size);
%load('theta.mat');
%save -text theta.mat Theta1 Theta2

%fprintf("Initial weights\n");
%fprintf("Theta1\n");
%fprintf("%f\n", Theta1);
%fprintf("\nTheta2\n");
%fprintf("%f\n", Theta2);
%fprintf("\n");

% Regularization factor
lambda = 0;

fprintf("Lambda: %f\n", lambda);

init_params = [Theta1(:) ; Theta2(:)];
J = cost(init_params, input_layer_size, hidden_layer_size, output_layer_size, Xtrain, ytrain, lambda);

fprintf('Initial cost: %f\n', J);

% Training
optopts = optimset('MaxIter', 25);
costFn = @(p) cost(p, ...
  input_layer_size, hidden_layer_size, output_layer_size, ...
  Xtrain, ytrain, lambda);

[params, J] = fmincg(costFn, init_params, optopts);
fprintf('Training iteration complete. Cost: %f\n', J);

% Measure error

[final_Theta1, final_Theta2] = paramMatrixify(params, input_layer_size, hidden_layer_size, output_layer_size);

[hTrain, errorTrain] = measureError(final_Theta1, final_Theta2, Xtrain, ytrain);
fprintf("Training classification error:     %f\n", errorTrain);

[hVal, errorVal] = measureError(final_Theta1, final_Theta2, Xval, yval);
fprintf("Validation classification error:   %f\n", errorVal);

[hTest, errorTest] = measureError(final_Theta1, final_Theta2, Xtest, ytest);
fprintf("Test classification error:         %f\n", errorTest);
