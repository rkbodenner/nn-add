function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. If z is a vector or matrix, returns
%   the gradient for each element.

g = sigmoid(z) .* (1 - sigmoid(z));

end
