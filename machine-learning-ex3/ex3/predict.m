function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the input data matrix (X)
X1 = [ones(m, 1) X];

% calculate h(x) of the first layer
t1 = sigmoid(X1*Theta1');

% calculate h(x) of the second layer

% Add ones to the input data matrix (t1)
X2 = [ones(m, 1) t1];
t2 = sigmoid(X2*Theta2');

% now find the max probablity of each row - get index, which is the class (0,1, 2, 3, 4, ... 9)
[w, p] = max(t2, [], 2);

% =========================================================================


end
