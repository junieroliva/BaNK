function BaNK_stats = run_bank_classification(X,Y,...
                        rks_cv_stats,trn_set,hol_set,tst_set,varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
nfreq = get_opt(opts, 'nfreq', rks_cv_stats.D);
lambdars = get_opt(opts, 'lambdars', rks_cv_stats.lambdars);
do_modev = get_opt(opts, 'do_modev', false);
lbfgsb_maxiter = get_opt(opts, 'lbfgsb_maxiter', 4000);
do_last_optimize = get_opt(opts, 'do_last_optimize', true);

if ~iscell(X)
    N = size(Y,1);
    d = size(X,2);
else
    N = length(X);
    d = size(X{1},2);
end
bagsizes = get_opt(opts, 'bagsizes');
do_mmd = iscell(X) | ~isempty(bagsizes);
if ~isempty(bagsizes)
    lastinds = cumsum(bagsizes);
end

% do BaNK <---------- FIX HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GMM prior
rho_prior = get_opt(opts, 'rho_prior', struct);
rho_prior.log_alpha = get_opt(rho_prior, 'log_alpha');
rho_prior.alpha = get_opt(rho_prior, 'alpha');
if isempty(rho_prior.alpha)
    rho_prior.alpha = 10;
else
    rho_prior.log_alpha = log(rho_prior.alpha);
end
rho_prior.K = get_opt(rho_prior, 'K', floor(nfreq/10));
rho_prior.nu = get_opt(rho_prior, 'nu', d+2);
rho_prior.kappa = get_opt(rho_prior, 'kappa', .1);
rho_prior.Psi = get_opt(rho_prior, 'Psi', eye(d)./rks_cv_stats.sigma2);
rho_prior.mu = get_opt(rho_prior, 'mu', zeros(d,1));

% sampling options 
sopts = get_opt(opts, 'sopts', struct);
sopts.burn_in = get_opt(sopts, 'burn_in', 20);
sopts.n_samp = get_opt(sopts, 'n_samp', 1);
sopts.samp_gap = get_opt(sopts, 'samp_gap', 1);
sopts.itprint = @it_bclass_print;

laplace_opts = get_opt(opts,'laplace_opts',struct('save_PhiW',do_modev));
init_Wstats = get_opt(opts,'init_Wstats');
if isempty(init_Wstats)
    init_Wstats = ... % this may make a substantial impact on performance
      @(data,nfreq,rho_prior,model_prior)...
      laplace_bin_class( ...
        struct('W',mvnrnd(zeros(1,d),rho_prior.Psi./(rho_prior.nu-d-1),nfreq)'),...
        data,model_prior,laplace_opts);
end

rks_lambda = rks_cv_stats.lambda;
lambda_mult = get_opt(opts, 'lambda_mult', 10.^(0:2:6));
nlambdars = length(lambdars);
nlmult = length(lambda_mult);

% update functions
if ~do_modev
    defupW = @(w,r,d,m)update_W_featwise_model_evidence(w,r,d,m,@laplace_w_beta_bin_class);
else
    lup = @(j,wjp,w,d,m)laplace_mod_ev_bin_class(j,wjp,w,d,m,laplace_opts);
    defupW = @(w,r,d,m)update_W_featwise_model_evidence(w,r,d,m,lup);
end
update_W = get_opt( opts, 'update_W', defupW);
update_rho = get_opt(opts, 'update_rho', @update_rho_no_shared_cov);
update.W = update_W;
update.rho = update_rho;

Ws = cell(nlmult,1);
rhos = cell(nlmult,1);
log_alphas = cell(nlmult,1);
betas = cell(nlambdars,nlmult);
hol_errs = nan(nlambdars,nlmult);
ala_stime = tic;
for lmi=1:nlmult
    model_prior.c = rks_lambda*lambda_mult(lmi);
    
    fprintf('## Running c = %g... \n',model_prior.c);
    drawnow('update');
    if ~do_mmd
        [Ws{lmi}, rhos{lmi}] = ...
          bank({X(trn_set,:), Y(trn_set)}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
    else
        if iscell(X)
            [Ws{lmi}, rhos{lmi}] = ...
              bank({X(trn_set), Y(trn_set)}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
        else
            [Ws{lmi}, rhos{lmi}] = ...
              bank({X(trn_set_bags,:), Y(trn_set), bagsizes(trn_set), cumsum(bagsizes(trn_set))}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
        end
    end
    
    log_alphas{lmi} = get_opt(rhos{lmi}{end},'log_alpha',rho_prior.log_alpha);
    W = Ws{lmi}(:,:,end);
    
    
    
    if ~do_mmd
        PhiW = X(trn_set,:)*W;
        PhiW = [cos(PhiW) sin(PhiW) ones(size(PhiW,1),1)];
        PhiWhol = X(hol_set,:)*W;
        PhiWhol = [cos(PhiWhol) sin(PhiWhol) ones(size(PhiWhol,1),1)];
    else
        if iscell(X)
            PhiW = cell2mat( cellfun(@(C)[mean([cos(C*W) sin(C*W)],1) 1],...
                X(trn_set), 'unif', false) );
            PhiWhol = cell2mat( cellfun(@(C)[mean([cos(C*W) sin(C*W)],1) 1],...
                X(hol_set), 'unif', false) );
        else
            PhiW = X*W; 
            PhiW = cumsum([cos(PhiW) sin(PhiW)],1);
            PhiW = PhiW(lastinds,:);
            PhiW = PhiW - [zeros(1,size(PhiW,2)); PhiW(1:end-1,:)];
            PhiW = bsxfun(@times,PhiW,1./bagsizes);
            PhiW = [PhiW ones(size(PhiW,1),1)];
            PhiWhol = PhiW(hol_set,:);
            PhiW = PhiW(trn_set,:);
        end
        
    end
    
    if do_last_optimize && ~do_mmd
        beta = zeros(size(PhiW,2),1);
        beta = newtons_binlr(PhiW,Y(trn_set,:),model_prior.c,beta);
        z = rhos{lmi}{end}.z;
        mus_z = rhos{lmi}{end}.mus;
        mus_z = reshape(mus_z,size(mus_z,1),1,size(mus_z,2));
        mus_z = mus_z(:,:,z);
        Siginvs_z = rhos{lmi}{end}.Sigmas;
        for l=1:size(Siginvs_z,3)
            Siginvs_z(:,:,l) = inv(Siginvs_z(:,:,l));
        end
        Siginvs_z = Siginvs_z(:,:,z);
        
        lb_cell = {-inf(nfreq,1), -inf(nfreq,1), -inf, -inf(d,nfreq)};
        ub_cell = { inf(nfreq,1),  inf(nfreq,1), inf, inf(d,nfreq)};
        auxdata = {X(trn_set,:),Y(trn_set,:),model_prior.c,mus_z,Siginvs_z};
        try
            [~, ~, ~, W] = ...
              lbfgsb( {beta(1:nfreq), beta(nfreq+1:end-1), beta(end), W}, lb_cell, ...
                ub_cell, 'rks_W_classification_obj','rks_W_classification_grad', auxdata,'print100callback', ...
                'm',25,'factr',1e-12, 'pgtol',1E-4,'maxiter',ceil(lbfgsb_maxiter/20) );
        catch
            try
                [~, ~, ~, W] = ...
                  lbfgsb( {beta(1:nfreq), beta(nfreq+1:end-1), beta(end), W}, lb_cell, ...
                    ub_cell, 'rks_W_classification_obj','rks_W_classification_grad', auxdata,'print100callback', ...
                    'm',40,'factr',1e-9, 'pgtol',1E-3,'maxiter',ceil(lbfgsb_maxiter/20) );
            catch
                warning('!! BaNK CV lbfgsb fail {c=%g} \n', model_prior.c);
            end
        end
        PhiW = X(trn_set,:)*W;
        PhiW = [cos(PhiW) sin(PhiW) ones(size(PhiW,1),1)];
        PhiWhol = X(hol_set,:)*W;
        PhiWhol = [cos(PhiWhol) sin(PhiWhol) ones(size(PhiWhol,1),1)];
        Ws{lmi}(:,:,end) =  W;
    end
    
    
    beta = zeros(size(PhiW,2),1);
    l_hol_errs = nan(nlambdars,1);
    for j=1:nlambdars
        beta = newtons_binlr(PhiW,Y(trn_set,:),lambdars(j),beta);
        Predict = PhiWhol*beta>0;
        l_hol_errs(j) = mean(Predict ~= Y(hol_set));
        betas{j,lmi} = beta;
    end
    hol_errs(:,lmi) = l_hol_errs;
    
    fprintf('## BaNK {c = %g} best hold ERR: %g \n', model_prior.c, min(l_hol_errs) );
end

[j,lmi] = find( hol_errs == min(hol_errs(:)) );
lmi = lmi(1);
j = j(1);

model_prior.c = rks_lambda*lambda_mult(lmi);
fprintf('## Final running c = %g... \n',model_prior.c);
org_burn_in = sopts.burn_in;
sopts.burn_in = get_opt(sopts, 'burn_in_last', 10*org_burn_in);
init_Wstats = ... % this may make a substantial impact on performance
      @(data,nfreq,rho_prior,model_prior)...
      laplace_bin_class( struct('W',Ws{lmi}(:,:,end)),data,model_prior,laplace_opts);
sopts.rho = rhos{lmi}{end};
rho_prior.log_alpha = log_alphas{lmi};
if ~do_mmd
    [Ws, rhos] = ...
          bank({X(trn_set|hol_set,:), Y(trn_set|hol_set)}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
else
    if iscell(X)
        [Ws, rhos] = ...
              bank({X(trn_set|hol_set), Y(trn_set|hol_set)}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
    else
        [Ws, rhos] = ...
          bank({X(trn_set_bags|hol_set_bags,:), Y(trn_set|hol_set), bagsizes(trn_set|hol_set), cumsum(bagsizes(trn_set|hol_set))},...
            nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
    end
end
W = Ws(:,:,end);

if ~do_mmd
    PhiW = X(trn_set|hol_set,:)*W;
    PhiW = [cos(PhiW) sin(PhiW) ones(size(PhiW,1),1)];
    PhiWtst = X(tst_set,:)*W;
    PhiWtst = [cos(PhiWtst) sin(PhiWtst) ones(size(PhiWtst,1),1)];
else
    if iscell(X)
        PhiW = cell2mat( cellfun(@(C)[mean([cos(C*W) sin(C*W)],1) 1],...
            X(trn_set|hol_set), 'unif', false) );
        PhiWtst = cell2mat( cellfun(@(C)[mean([cos(C*W) sin(C*W)],1) 1],...
            X(tst_set), 'unif', false) );
    else
        PhiW = X*W; 
        PhiW = cumsum([cos(PhiW) sin(PhiW)],1);
        PhiW = PhiW(lastinds,:);
        PhiW = PhiW - [zeros(1,size(PhiW,2)); PhiW(1:end-1,:)];
        PhiW = bsxfun(@times,PhiW,1./bagsizes);
        PhiW = [PhiW ones(size(PhiW,1),1)];
        PhiWtst = PhiW(tst_set,:);
        PhiW = PhiW(trn_set|hol_set,:);
    end
end

if do_last_optimize && ~do_mmd
    beta = zeros(size(PhiW,2),1);
    beta = newtons_binlr(PhiW,Y(~tst_set,:),model_prior.c,beta);
    z = rhos{end}.z;
    mus_z = rhos{end}.mus;
    mus_z = reshape(mus_z,size(mus_z,1),1,size(mus_z,2));
    mus_z = mus_z(:,:,z);
    Siginvs_z = rhos{end}.Sigmas;
    for l=1:size(Siginvs_z,3)
        Siginvs_z(:,:,l) = inv(Siginvs_z(:,:,l));
    end
    Siginvs_z = Siginvs_z(:,:,z);

    lb_cell = {-inf(nfreq,1), -inf(nfreq,1), -inf, -inf(d,nfreq)};
    ub_cell = { inf(nfreq,1),  inf(nfreq,1), inf, inf(d,nfreq)};
    auxdata = {X(~tst_set,:),Y(~tst_set,:),model_prior.c,mus_z,Siginvs_z};
    try
        [~, ~, ~, W] = ...
          lbfgsb( {beta(1:nfreq), beta(nfreq+1:end-1), beta(end), W}, lb_cell, ...
            ub_cell, 'rks_W_classification_obj','rks_W_classification_grad', auxdata,'print100callback', ...
            'm',25,'factr',1e-12, 'pgtol',1E-4,'maxiter',ceil(lbfgsb_maxiter/10) );
    catch
        try
            [~, ~, ~, W] = ...
              lbfgsb( {beta(1:nfreq), beta(nfreq+1:end-1), beta(end), W}, lb_cell, ...
                ub_cell, 'rks_W_classification_obj','rks_W_classification_grad', auxdata,'print100callback', ...
                'm',40,'factr',1e-9, 'pgtol',1E-3,'maxiter',ceil(lbfgsb_maxiter/10) );
        catch
            warning('!! BaNK CV lbfgsb fail {c=%g} \n', model_prior.c);
        end
    end
    PhiW = X(~tst_set,:)*W;
    PhiW = [cos(PhiW) sin(PhiW) ones(size(PhiW,1),1)];
    PhiWtst = X(tst_set,:)*W;
    PhiWtst = [cos(PhiWtst) sin(PhiWtst) ones(size(PhiWtst,1),1)];
end


beta = newtons_binlr(PhiW,Y(trn_set|hol_set,:),lambdars(j),betas{j,lmi});
Predict = PhiWtst*beta>0;



BaNK_stats.Y_pred = Predict;
BaNK_stats.resids =  (Predict ~= Y(tst_set));
BaNK_stats.tst_err = mean(Predict ~= Y(tst_set));
BaNK_stats.hol_mses = hol_errs;
BaNK_stats.lambdars = lambdars;
BaNK_stats.lambda_mult = lambda_mult;
BaNK_stats.W = W;
BaNK_stats.beta = beta;
BaNK_stats.time = toc(ala_stime);
fprintf('## BaNK Test ERR: %g \n', BaNK_stats.tst_err);
end