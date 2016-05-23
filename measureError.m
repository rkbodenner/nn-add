function [h,error] = measureError(Theta1, Theta2, X, y)
% MEASUREERROR predicts outputs given parameters and inputs for a two-layer
% neural network. Returns the outputs and the error over the entire input.

h = predict(Theta1, Theta2, X);

error = sum(h' != y) / size(y,1);
