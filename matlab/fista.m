function x = fista(A, y, lambda, max_iter, tol)
    % FISTA for L1-regularized least squares with constant stepsize
    % Algorithm described https://www.ceremade.dauphine.fr/~carlier/FISTA
    % Pg. 11
    % Args:
    %   A         : Sensing matrix
    %   y         : Observed measurements
    %   lambda    : Regularization parameter for sparsity
    %   max_iter  : Maximum number of iterations
    %   tol       : Convergence tolerance
    % Returns:
    %   x         : Reconstructed sparse signal

    % Initialize variables
    [~, n] = size(A);
    x = zeros(n, 1);  % Initial guess for the solution
    z = x;            % Auxiliary variable
    t = 1.0;          % Step size parameter
    L = norm(A, 2)^2; % Estimate of Lipschitz constant
    alpha = 1 / L;    % Step size for gradient descent
    iter = 0;         % Iteration counter

    for k = 1:max_iter
        % Save previous x for convergence check
        iter = iter + 1;
        %disp(iter);
        x_old = x;

        % Gradient step
        grad = A' * (A * z - y);  % Compute gradient of the data term
        x = soft_thresholding(z - alpha * grad, lambda * alpha);

        % Update momentum term
        t_new = (1 + sqrt(1 + 4 * t^2)) / 2;
        z = x + ((t - 1) / t_new) * (x - x_old);

        % Update t for next iteration
        t = t_new;

        % Stopping criterion (convergence check)
        if norm(x - x_old) < tol
            break;
        end
    end
end

function x = soft_thresholding(x, threshold)
    % Soft-thresholding operator for L1 regularization (Described Pg. 7, eq. 2.6)
    x = sign(x) .* max(abs(x) - threshold, 0);
end
