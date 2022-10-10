function [results] = svm_admm(trainA, lambda, Kmax, p, rho)
% Start stopwatch to measure execution time
t_start = tic; % Timer start

% Set tolerances for stop condition
tolAbs   = 1e-4;
tolRel   = 1e-2;

Dataset = trainA;
% We simulate a distributed consensus in a serial way so the data must be treated appropriately
n = size(Dataset,2); %extract columns number
N = max(p); % retrieve numbers of partitions

% Partition samples and assign them to each agent
temp = cell(1,N); % Preallocate variable to reduce computational time
for i = 1:N
    temp{i} = Dataset(p==i,:);
end
Dataset = temp;

% Setup and use ADMM solver
% Preallocate variables to reduce computational time
x = randn(n,N);
z = randn(n,N);
u = randn(n,N);

fprintf('%3s\t%8s\t%8s\t%8s\t%8s\n', 'iter', 'r_norm', 'eps_pri', 's_norm', 'eps_dual');
for k = 1:Kmax % Iteration's steps
    % we need to minimize   (1/2)||w||_2^2 + \lambda sum h_j(w, b)
    % simulate multiple agents, updating xi
    for i = 1:N % parfor could be used if a parallel pool processes is available
        x(:,i) = CVXOptim(Dataset,n, i, rho, z, u); % save updated xi ( save x_var in i column of x)
    end

    % z-update
    zold = z;
    z = N*rho*(1/(2*lambda + N*rho))*(mean(x, 2)+ mean(u,2));
    z(size(z,1),:) = ((2*lambda+N*rho)/N*rho)*z(size(z,1),:);
    z = z*ones(1,N);

    % u-update
    parfor i = 1:N
        u(:,i)=u(:,i) + x(:,i) - z(:,i);
    end

    % Save and return steps values used to draw plots
    results.r_norm(k)  = norm(x - z); % L2 norm of x-z
    results.s_norm(k)  = norm(-rho*(z - zold));
    results.eps_pri(k) = sqrt(n)*tolAbs + tolRel*max(norm(x), norm(-z));
    results.eps_dual(k)= sqrt(n)*tolAbs + tolRel*norm(rho*u);
    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n', k, results.r_norm(k), results.eps_pri(k), results.s_norm(k), results.eps_dual(k));

    % Save and then return the optimized parameters to plot data and decision boundary
    results.lastx = x;

    %check terminations conditions and stop if needed
    if (results.r_norm(k) < results.eps_pri(k) && results.s_norm(k) < results.eps_dual(k))
        break;
    end
end
end

% Optimization function leveraging CVX problem solver.
% Declared as a function it could be parallelized using parfor on processes (not on threads)
function x_var = CVXOptim(Dataset, n, i, rho, z, u)
cvx_begin quiet % using CVX to solve the convex minimization problem
variable x_var(n) % varibale xi to be computed and updated
minimize ( sum(pos(Dataset{i}*x_var + 1)) + (rho/2)*sum_square(x_var - z(:,i) + u(:,i)) ) % pos indicates the positive part (performs the max between (0 and f)
cvx_end
end
