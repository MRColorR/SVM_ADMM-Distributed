function [results] = svm_admm(trainSamples, trainLabels, lambda, Kmax, p, rho)
% Start stopwatch to measure execution time
t_start = tic;

% Set tolerances for stop condition
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
Dataset = [trainSamples,trainLabels];
% We simulate a distributed consensus in a serial way so the data must be treated appropriately
n = size(Dataset,2); %extract columns dimension 
N = max(p); % retrieve numbers of partitions

% group samples together
tmp = cell(1,N); % Preallocate variable to reduce computational time
for i = 1:N
    tmp{i} = Dataset(p==i,:);
end
Dataset = tmp;

% Setup and use ADMM solver
% Preallocate variables to reduce computational time
x = zeros(n,N);
z = zeros(n,N);
u = zeros(n,N);

fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', 'r norm', 'eps pri', 's norm', 'eps dual', 'objective');

for k = 1:Kmax % Iteration's steps
    % we need to minimize   (1/2)||w||_2^2 + \lambda sum h_j(w, b)
    % simulate multiple agents, updating xi
    for i = 1:N
        cvx_begin quiet % using CVX to solve the convex minimization problem
        variable x_var(n) % varibale xi to be computed and updated
        minimize ( sum(pos(Dataset{i}*x_var + 1)) + rho/2*sum_square(x_var - z(:,i) + u(:,i)) )
        cvx_end
        x(:,i) = x_var; % save updated xi ( save x_var in i column of x)
    end


    % z-update
    zold = z;
    z = N*rho/(1/lambda + N*rho)*mean( x + u, 2 );
    z = z*ones(1,N);

    % u-update
    u = u + (x - z);

    % diagnostics, reporting, termination checks
    results.objval(k)  = objective(Dataset, lambda, x, z);

    %================


    results.r_norm(k)  = norm(x - z); % L2 norm of x-z
    results.s_norm(k)  = norm(-rho*(z - zold));

    results.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    results.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, results.r_norm(k), results.eps_pri(k), results.s_norm(k), results.eps_dual(k), results.objval(k));
    %
    % THIS PART IS ONLY AN ATTEMPT TO GET THE OPTIMIZED PARAMETERS TO DRAW THE DATA AND THE SEPARATOR HYPERPIAN OF THE TRAINED SVM. FOR NOW I HAVE NOT SUCCESS ...
    results.lastx = x;
    results.lastz = z;
    results.lastu = u;

    if (results.r_norm(k) < results.eps_pri(k) && results.s_norm(k) < results.eps_dual(k))
        break;
    end
end


toc(t_start);

end

function obj = objective(Dataset, lambda, x, z)
obj = hinge_loss(Dataset,x) + 1/(2*lambda)*sum_square(z(:,1));
end

function val = hinge_loss(Dataset,x)
val = 0;
for i = 1:length(Dataset)
    val = val + sum(pos(Dataset{i}*x(:,i) + 1));
end
end