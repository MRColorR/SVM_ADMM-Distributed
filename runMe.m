% Clear all if needed
clear all;
close all;
clc;

% Parameters setup

lambda = 1.0; % set lambda of the SVM fittig. Lambda is the regularization parameter of the SVM that multiply the hinge loss function

Kmax = 500; % Maximum number of iteration indicates the acceptable value within which convergence should be achieved to avoid forced stop. Should be more than enough to reach convergence.

rho = 1.0; % Set rho parameter of the augmented Lagrangian. It is part of the regularization term added for obtaining a strictly convex optimization problem

[Dataset,p] = newData("load"); % if argument="load" it loads, else for any other argument e.g. "Random" it generates new data collected in Dataset. p indicates the partitioned sets


% Let's call the function to solve the SVM problem using ADMM
[results] = svm_admm(Dataset, lambda, Kmax, p, rho);

% Now let's see the results
K = length(results.objval); % Retrieve all steps' data

% Plot the resulting values for each iteration
figure;
plot(1:K, results.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

% Plot performance trend as the iterations pass
figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, results.r_norm), 'k', 1:K, results.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, results.s_norm), 'k', 1:K, results.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');
%
% Plot data THIS PART IS ONLY AN ATTEMPT TO DRAW THE DATA AND THE SEPARATOR HYPERPIAN. FOR NOW I HAVE NOT SUCCESS ...
%
figure;
hold on;
gscatter(Dataset(:,1),Dataset(:,2), Dataset(:,size(Dataset,2)));

xavg = mean(results.lastx,2);
zavg = mean(results.lastz,2);
plot (zavg*xavg',xavg);
hold off;
%THE PART OF THE CLASSIFICATION TEST ON DATA NOT SEEN BEFORE IS ALSO MISSING. (ONE THING AT A TIME)