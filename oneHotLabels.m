function Y = oneHotLabels(output_layer_size, y)
% ONEHOTLABELS converts output values y to a matrix of
% one-hot column vectors whose ith value is hot (==1) when the result
% value+1 == i. For example, if a result is 42, the label vector Y(:,i) is:
% [0; 0; ... 1; 0; 0] where the 1 is at Y(43,i).

Y = zeros(output_layer_size, size(y));
for i = 1:size(y)
  % 1st element represents the value 0, 2nd 1, etc.
  Y(y(i)+1, i) = 1;
endfor
