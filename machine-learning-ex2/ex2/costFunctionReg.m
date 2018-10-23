function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% this works - original way
%J1 = -(1/m)*sum(y'*log(sigmoid(X*theta)) + (1 - y)'*(log(1-sigmoid(X*theta))));
%theta1 = theta(2:length(theta));
%J2 = (lambda/(2*m))*sum(theta1'*theta1);
%J = J1 + J2;

% gradient calculations - orignal way 
%grad(1) = 1/m*sum((sigmoid(X*theta) - y)' * X(:,1));

%n = length(theta);
%for i = 2:n
 % grad(i) = 1/m*sum((sigmoid(X*theta) - y)' * X(:,i)) + (lambda/m)*theta(i);
%end

% % vectorized implementation
temp = theta;
temp(1) = 0;

%J1 = -(1/m)*sum(y'*log(sigmoid(X*theta)) + (1 - y)'*(log(1-sigmoid(X*theta))));
%J2 = (lambda/(2*m))*sum(temp'*temp);
%J = J1 + J2;

J1 = (lambda/(2*m))*sum(temp'*temp);
J2 = (lambda/(2*m))*(temp'*temp);

J = -(1/m)*sum(y'*log(sigmoid(X*theta)) + (1 - y)'*(log(1-sigmoid(X*theta)))) + (lambda/(2*m))*sum(temp'*temp); 

Reg = (lambda/(2*m))*sum(temp'*temp);
fprintf('Regularized Parameters %f\n', Reg);


grad = (1/m*((sigmoid(X*theta))-y)'*X)'+ (lambda/m)*temp;

fprintf('grad: %f\n', grad);
%fprintf('  theta: %f\n', theta);


% =============================================================

end
