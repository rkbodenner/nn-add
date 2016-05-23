function h = outputFromOneHotLabels(hvec)
% OUTPUTFROMONEHOTLABELS converts a one-hot vector of labels to an output value.

% Find the index of the strongest activation among the output nodes. This is the predicted result's value - 1.
[dummy, p] = max(hvec);
% 1st element represents the value 0, 2nd 1, etc.
h = p - 1;
