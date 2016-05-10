function method_stats = run_regression(X,Y,varargin)
% run_regression Run BaNK for regression models. This function first runs
% ridge regression on random kitchen sink features for a RBF kernel (i.e.
% drawing frequencies from a Gaussian validating based on MSE). After, the
% function runs BaNK for regression.
% Inputs:
%   X - N x d covariate matrix of inputs
%   Y - N x 1 response vector of outputs (TODO: test for multi-d response)
%   opts (optional) - (TODO: write description)
% Outputs: (TODO: write)
% TODO: add capability for running additional functions?

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end


funcs = get_opt(opts, 'funcs', []);
method_stats = cell(length(funcs)+1,1);

% options
rks_opts = get_opt(opts, 'rks_opts', struct);
do_quant_sigma2s = get_opt(opts, 'do_quant_sigma2s', true);
nfreq = get_opt(opts, 'nfreq', 384);

% split data
[N,d] = size(X);
N_rot = get_opt(opts, 'N_rot', min(2000,N));
trn_set = get_opt(opts, 'trn_set');
hol_set = get_opt(opts, 'hol_set');
tst_set = get_opt(opts, 'tst_set');
if isempty(trn_set) || isempty(hol_set) || isempty(tst_set) 
    tprec = get_opt(opts, 'tprec', .1);
    hprec = get_opt(opts, 'hprec', .1);
    [trn_set, hol_set, tst_set] = split_data( N, tprec, hprec );
end

rks_stime = tic;
% get rule of thumb bandwidths
if do_quant_sigma2s
    pd2s = dists2(X(randperm(N,N_rot),:),X(randperm(N,N_rot),:));
    rks_opts.sigma2s = quantile(pd2s(:), .1:.2:.9);
end
rks_opts.trn_set = trn_set;
rks_opts.hol_set = hol_set;
rks_opts.tst_set = tst_set;

% get RKS ridge regression
lambdars = get_opt(opts,'lambdars',2.^(10:-1:-10));
rks_opts.verbose = true;
rks_opts.do_sin = true;
rks_opts.D = nfreq;
rks_opts.lambdas = lambdars;
[B, rks, RKS_stats, rks_cv_stats] = rks_ridge(X,Y,rks_opts);

rks_cv_stats.B = B;
rks_cv_stats.rks = rks;
rks_cv_stats.D = nfreq;
rks_cv_stats.lambdars = lambdars;

RKS_stats.tst_err = RKS_stats.mse;
RKS_stats.resids = RKS_stats.pred - Y(tst_set);

RKS_stats.cv_stats = rks_cv_stats;
RKS_stats.time = toc(rks_stime);
method_stats{1} = RKS_stats;

for i=1:length(funcs)
    method_stats{i+1} = funcs{i}(X,Y,rks_cv_stats,trn_set,hol_set,tst_set);
end


end