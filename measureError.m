function [h,error] = measureError(Theta1, Theta2, X, y)
% MEASUREERROR predicts outputs given parameters and inputs for a two-layer
% neural network. Returns the outputs and the error over the entire input.

h = predict(Theta1, Theta2, X);

% Map outputs to the single feature: Is the output of the target value?
y = y == 2;

error = sum(h' != y) / size(y,1);
