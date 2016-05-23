% Neural network for learning to add pairs of numbers 50 > x >= 0

% Load training examples, created with:
%
%X = floor(rand(10000,2) * 50);
%y = X(:,1) + X(:,2);
%save -text training-examples.mat X y
load('training-examples.mat');

m = size(X,1);
mtrain = 0.6 * m;
mval = 0.2 * m;
mtest = m - mtrain - mval;
Xtrain = X(1:mtrain,:);
Xval = X(mtrain+1:mtrain+mval,:);
Xtest = X(mtrain+mval+1:m,:);
ytrain = Xtrain(:,1) + Xtrain(:,2);
yval = Xval(:,1) + Xval(:,2);
ytest = Xtest(:,1) + Xtest(:,2);

input_layer_size = 2;
hidden_layer_size = 128;
output_layer_size = 99;  % One label for each possible integer result value

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
lambda = 1.0;

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

[final_Theta1, final_Theta2] = paramMatrixify(params, input_layer_size, hidden_layer_size, output_layer_size);

h = predict(final_Theta1, final_Theta2, Xtrain);
error = sum(h' != ytrain) / size(ytrain,1);
fprintf("Training classification error: %f\n", error);

h = predict(final_Theta1, final_Theta2, Xval);
error = sum(h' != yval) / size(yval,1);
fprintf("Validation classification error: %f\n", error);

h = predict(final_Theta1, final_Theta2, Xtest);
error = sum(h' != ytest) / size(ytest,1);
fprintf("Test classification error: %f\n", error);
