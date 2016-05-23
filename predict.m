function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained two-layer neural network

m = size(X, 1);

a1 = [ones(m, 1) X];
z2 = Theta1 * a1';
a2 = [ones(1, m); sigmoid(z2)];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

p = outputFromOneHotLabels(a3);
