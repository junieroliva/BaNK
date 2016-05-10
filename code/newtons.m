function [x, obj] = newtons(x, g, varargin)
%newtons runs newton's method on the input function
% g = {@obj, @grad, @hess} Cell array containing objective, gradient and
% Hessian of function to be optimized
% optional arguments:
% options, a struct with the following fields:
% beta = backtracking parameter
% maxIT = maximum number of iterations
% tol = convergence criterion


if isempty(varargin)
    options = struct;
else
    options = varargin{1};
end
if ~isfield(options, 'beta')
    beta = 0.8;
else
    beta = options.beta;
end
if ~isfield(options, 'maxIT')
    maxIT = 100;
else
    maxIT = options.maxIT;
end
if ~isfield(options, 'tol')
    tol = 1e-7;
else
    tol = options.tol;
end
if ~isfield(options, 'verbose')
    verbose = false;
else
    verbose = true;
end
    
it = 1;
obj = nan(maxIT, 1);
obj(1) = g{1}(x);
if verbose
    fprintf('\t[it:%g] obj=%f\n',it,obj(it));
end
while ~isConverged(obj, it, tol) && it < maxIT
    % get gradient
    grad = g{2}(x);
    % compute direction
    if length(g)>3
        H_inv = g{4}(x);
        d = -H_inv*grad;
    else
        H = g{3}(x);
        d = -H\grad;
    end
    % backtrack
    t = 1;
    objx = g{1}(x);
    while g{1}(x + t*d) > objx + 0.5*t*dot(grad, d)
        t = beta*t;
    end
    % update
    x = x + t*d;
    it = it + 1;
    obj(it) =  g{1}(x);
    if verbose
        fprintf('\t[it:%g] obj=%f\n',it,obj(it));
    end
end
obj = obj(1:(it-1));
end

function conv = isConverged(obj, it, tol)
if it == 1 
    conv = false;
else
    conv = abs((obj(it) - obj(it-1))/obj(it-1)) < tol;
end
end