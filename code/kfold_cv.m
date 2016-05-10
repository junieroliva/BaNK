function [tst_errs, tst_resids, method_stats, inds, hol_inds] = ...
    kfold_cv(X,Y,runfunc,funcs,varargin)

opts = struct;
if ~isempty(varargin)
    opts = varargin{1};
end

stand_response = get_opt(opts, 'stand_response', true);
if stand_response
    Y = (Y-mean(Y))./std(Y);
end

fileprefix = get_opt(opts, 'fileprefix');

N = size(X,1);
nfuncs = length(funcs)+1;
names =  get_opt(opts, 'names', []);
do_names = ~isempty(names) && length(names)==nfuncs;

if exist('gcp', 'file')
    p = gcp('nocreate');
else
    p = [];
end
if ~isempty(p) && p.NumWorkers>1
    Kdef = p.NumWorkers;
else
    Kdef = 5;
end
K = get_opt(opts, 'K', Kdef);
inds = get_opt(opts, 'inds', crossvalind('Kfold', N, K));
hol_inds = get_opt(opts, 'hol_inds');
if isempty(hol_inds)
    hol_inds = nan(K,1);
    for k=1:K
        k_hol = [1:(k-1) (k+1):K];
        k_hol = k_hol(randi(K-1));
        hol_inds(k) = k_hol;
    end
end

Krun = get_opt(opts, 'Krun', K);
tst_errs = nan(Krun,nfuncs);
tst_resids = cell(Krun,nfuncs);
method_stats = cell(Krun,nfuncs);
parfor k=1:Krun
    k_hol = hol_inds(k);
    run_exp_opts = get_opt(opts, 'run_exp_opts', struct);
    run_exp_opts.hol_set = inds==k_hol;
    run_exp_opts.tst_set = inds==k;
    run_exp_opts.trn_set = inds~=k & inds~=k_hol;
    run_exp_opts.funcs = funcs;
    %run_exp_opts.tst_set
    k_method_stats = runfunc(X,Y,run_exp_opts);
    
    tst_errs_k = nan(nfuncs,1);
    tst_resids_k = cell(nfuncs,1);
    method_stats_k = cell(nfuncs,1);
    for fi=1:nfuncs
        %k_method_stats{fi}
        tst_errs_k(fi) = k_method_stats{fi}.tst_err;
        tst_resids_k{fi} = k_method_stats{fi}.resids;
        method_stats_k{fi} = k_method_stats{fi};
    end
    tst_errs(k,:) = tst_errs_k;
    tst_resids(k,:) = tst_resids_k;
    method_stats(k,:) = method_stats_k;
    
    fprintf('\n\n\n\n\n');
    fprintf('_________________________________________________________\n');
    fprintf('K: %i K_hol: %i\n',k,k_hol);
    for fi=1:nfuncs
        if do_names
            fprintf('%s err:\t%g\n',names{fi},tst_errs_k(fi));
        else
            fprintf('Method_%g err:\t%g\n',fi,tst_errs_k(fi));
        end
    end
    fprintf('_________________________________________________________\n');
    fprintf('\n\n\n\n\n');
    
    if ~isempty(fileprefix)
        parsave(sprintf('%s_%i_tst_errs_k.mat',fileprefix,k),tst_errs_k);
        parsave(sprintf('%s_%i_tst_resids_k.mat',fileprefix,k),tst_resids_k);
        parsave(sprintf('%s_%i_method_stats_k.mat',fileprefix,k),method_stats_k);
    end
end

tst_resids = cell2mat(tst_resids);

if ~isempty(fileprefix)
    save(sprintf('%s.mat',fileprefix), ...
         'tst_errs', 'tst_resids', 'method_stats', 'inds', 'hol_inds');
end

end