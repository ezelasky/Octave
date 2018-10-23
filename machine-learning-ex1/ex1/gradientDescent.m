function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    theta1_temp = theta(1) - (1/m)* alpha * sum(((X*theta) - y)' * X(:,1));
     
     theta2_temp = theta(2) - (1/m)* alpha * sum(((X*theta) - y)' * X(:,2));
     
     theta(1) = theta1_temp;
     theta(2) = theta2_temp;
  
    %vectorized approach - from student code - it works
    %h=X*theta;
    %errors=h-y;
    %theta_change= alpha*1/m*(X'*errors);
    %theta = theta - theta_change;
  
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    fprintf('Theta found by gradient descent: iter: %f \n', iter);
    fprintf('%f %f ', theta(1), theta(2));
    fprintf('  %f \n', J_history(iter));

end

end
