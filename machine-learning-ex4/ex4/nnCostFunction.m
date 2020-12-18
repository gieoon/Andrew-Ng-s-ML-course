function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
size(X)
% Add column of ones to the start of X
X = [ones(m, 1) X];
%size(X) % 5000 X 401
%size(Theta1) % 25 X 401
%size(Theta2) % 10 X 26
z1 = X * Theta1'; % Multiplying these two gives a matrix of all the activations after combining the weights for a single layer
a1 = sigmoid(z1);
a1 = [ones(m,1) a1];
z2 = a1 * Theta2';
h = sigmoid(z2);
%size(h(:,1)) % size(h) = 5000,10 (10 one-hot vector for each example)
%num_labels
%Theta1(:,1)
% Loop through all examples and calculate the cost of each
% Loop through each one-hot vector column
for k=1:num_labels
    yk = y == k;
    %y % list of each label 5000 X 1
    %y==k % 1 if the label is the same as the row number (one-hot is switched on)

    %size(yk)
    % h has one-hot vectors horizontally, so we judge whether hypothesis is correct or not for each one-hot vector
    J = J + ( sum( (-yk .* log(h(:,k))) - ((1 - yk) .* log(1-h(:,k))) )  / m);
end

regTheta1 = sum(sum(Theta1(:,2:end).^2));
regTheta2 = sum(sum(Theta2(:,2:end).^2));
reg = (lambda * (regTheta1 + regTheta2)) / (2 * m);
J = J + reg;

%c1 = sum( (-y .* log(h1)) - ((1 - y) .* log(1-h1)) ) / m;
%reg = (lambda / (2 * m)) .* sum(Theta1(2:end).^2);
%J1 = c1 + reg; 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

d2 = 0;
d1 = 0;
for t=1:m
    A1 = X(t,:)';
    Z2 = Theta1 * A1;
    A2 = [1; sigmoid(Z2)];
    Z3 = Theta2 * A2;
    A3 = sigmoid(Z3);

    % yt = y(t,:); % y for current example
    % delta_3 = a_3 - yt;
    yk = zeros(num_labels,1);
    yk(y(t,:)) = 1; % Sets the column to 1 where the class is the same
    delta_3 = A3 - yk;

    delta_2 = Theta2(:,2:end)' * delta_3 .* sigmoidGradient(Z2);
    %size(delta_2)
    %size(A1)
    d1 = d1 + delta_2 * A1';
    d2 = d2 + delta_3 * A2';
end

Theta1_grad = d1 / m + (lambda/m) * [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad = d2 / m + (lambda/m) * [zeros(num_labels, 1) Theta2(:,2:end)];

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
