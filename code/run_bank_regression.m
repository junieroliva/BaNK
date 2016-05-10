function BaNK_stats = run_bank_regression(X,Y,...
                        rks_cv_stats,trn_set,hol_set,tst_set,varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

[N,d] = size(X);
N_trn = sum(trn_set);
N_tst = sum(tst_set);
YtY = Y(trn_set)'*Y(trn_set);

% options
nfreq = get_opt(opts, 'nfreq', rks_cv_stats.D);
lambdars = get_opt(opts, 'lambdars', rks_cv_stats.lambdars);
lbfgsb_maxiter = get_opt(opts, 'lbfgsb_maxiter', 4000);
do_last_optimize = get_opt(opts, 'do_last_optimize', true);
use_sigma2_error = get_opt(opts, 'use_sigma2_error', true);

% update functions
update_W = get_opt(opts, 'update_W', ...
  @(w,r,d,m)update_W_featwise_model_evidence(w,r,d,m,@lowrank_update_W_stats_reggression));
update_rho = get_opt(opts, 'update_rho', @update_rho_no_shared_cov);
update.W = update_W;
update.rho = update_rho;

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
rho_prior.nu = get_opt(rho_prior, 'nu', size(X,2)+2);
rho_prior.kappa = get_opt(rho_prior, 'kappa', .1);
rho_prior.Psi = get_opt(rho_prior, 'Psi', eye(size(X,2))/rks_cv_stats.sigma2);
rho_prior.mu = get_opt(rho_prior, 'mu', zeros(size(X,2),1));

% sampling options 
sopts = get_opt(opts, 'sopts', struct);
sopts.burn_in = get_opt(sopts, 'burn_in', 25);
sopts.n_samp = get_opt(sopts, 'n_samp', 1);
sopts.samp_gap = get_opt(sopts, 'samp_gap', 1);
sopts.itprint = get_opt(sopts, 'itprint', @it_reg_print);

init_Wstats = get_opt(opts,'init_Wstats');
if isempty(init_Wstats)
    init_Wstats = ... % this may make a substantial impact on performance
      @(data,nfreq,rho_prior,model_prior)...
      W_stats_reg_model_ev( ...
        mvnrnd(zeros(1,size(data{1},2)),...
        rho_prior.Psi./(rho_prior.nu-size(data{1},2)-1),nfreq)',...
        data{1},data{2},model_prior,get_opt(opts,'init_opts',struct));
end

lambda_mult = get_opt(opts, 'lambda_mult', 10.^(0:2:6));
nlambdars = length(lambdars);
nlmult = length(lambda_mult);
Ws = cell(nlmult,1);
rhos = cell(nlmult,1);
log_alphas = cell(nlmult,1);
hol_mses = nan(nlambdars,nlmult);
bank_stime = tic;
for lmi=1:nlmult
    model_prior = struct;
    model_prior.a = 1;
    model_prior.b = 1;
    model_prior.c = lambda_mult(lmi)*rks_cv_stats.lambda;
    
    fprintf('## Running c = %g... \n',model_prior.c);
    drawnow('update');
    [Ws{lmi}, rhos{lmi}] = ...
      bank({X(trn_set,:), Y(trn_set,:)}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
    log_alphas{lmi} = get_opt(rhos{lmi}{end},'log_alpha',rho_prior.log_alpha);
    W = Ws{lmi}(:,:,end);
    PhiW_all = [cos(X*W) sin(X*W)];
    PhiW = PhiW_all(trn_set,:);
    PtP = PhiW'*PhiW;
    PtY = PhiW'*Y(trn_set,:);
    
    if do_last_optimize
        beta = (PtP+model_prior.c*eye(2*nfreq))\(PtY);
        
        % use mode of posterior inv-gamma for noise variance
        if use_sigma2_error
            a_n = model_prior.a + N_trn/2.0;
            b_n = model_prior.b + 0.5* ( YtY - PtY'*beta );
            sigma2_error = b_n/(a_n+1);
        else
            sigma2_error = 1;
        end
        fprintf('## BaNK {c=%g} sigma2_error: %g \n', model_prior.c, sigma2_error);
        
        z = rhos{lmi}{end}.z;
        mus_z = rhos{lmi}{end}.mus;
        mus_z = reshape(mus_z,size(mus_z,1),1,size(mus_z,2));
        mus_z = mus_z(:,:,z);
        Siginvs_z = rhos{lmi}{end}.Sigmas;
        for l=1:size(Siginvs_z,3)
            Siginvs_z(:,:,l) = inv(Siginvs_z(:,:,l));
        end
        Siginvs_z = Siginvs_z(:,:,z);
        
        % lbfgsb version
%         lb_cell = {-inf(nfreq,1), -inf(nfreq,1), -inf(d,nfreq)};
%         ub_cell = { inf(nfreq,1),  inf(nfreq,1),  inf(d,nfreq)};
        auxdata = {X(trn_set,:),Y(trn_set,:),model_prior.c,mus_z,Siginvs_z,sigma2_error};
%         [~, ~, W] = ...
%           lbfgsb( {beta(1:nfreq), beta(nfreq+1:end), W}, lb_cell, ...
%             ub_cell, 'rks_W_obj','rks_W_grad', auxdata,'print100callback', ...
%             'm',25,'factr',1e-12, 'pgtol',1E-4,'maxiter',ceil(lbfgsb_maxiter/20) );
        % minFunc version %TODO: check
        optfunc = @(vars)rks_W_minfunc(vars,auxdata);
        opts_mf = struct;
        opts_mf.maxFunEvals = ceil(lbfgsb_maxiter/20);
        opts_mf.Display = 'off';
        opts_mf.Method = 'lbfgs';
        vars = minFunc(optfunc, [beta(1:nfreq); beta(nfreq+1:end); W(:)], opts_mf);
        W = reshape(vars(2*nfreq+1:2*nfreq+d*nfreq),d,nfreq);

        PhiW_all = [cos(X*W) sin(X*W)];
        PhiW = PhiW_all(trn_set,:);
        PtP = PhiW'*PhiW;
        PtY = PhiW'*Y(trn_set,:);
        Ws{lmi} = W;
    end
    
    l_hol_mses = nan(nlambdars,1);
    for li=1:nlambdars
        beta = (PtP+lambdars(li)*eye(2*nfreq))\(PtY);
        Y_pred = PhiW_all(hol_set,:)*beta;
        l_hol_mses(li) = mean(mean((Y(hol_set,:)-Y_pred).^2));
    end
    hol_mses(:,lmi) = l_hol_mses;
    fprintf('## BaNK {c=%g} best hol_mse: %g \n', model_prior.c, min(l_hol_mses));
end

[li,lmi] = find(hol_mses==min(hol_mses(:)));
model_prior = struct;
model_prior.a = 1;
model_prior.b = 1;
model_prior.c = lambda_mult(lmi)*rks_cv_stats.lambda;

W = Ws{lmi}(:,:,end);
PhiW_all = [cos(X*W) sin(X*W)];
PhiW = PhiW_all(~tst_set,:);
beta = (PhiW'*PhiW+lambdars(li)*eye(2*nfreq))\(PhiW'*Y(~tst_set,:));
Y_pred = PhiW_all(tst_set,:)*beta;
BaNK_stats.tst_mse_pre = mean(mean((Y(tst_set,:)-Y_pred).^2));
BaNK_stats.Y_pred_pre = Y_pred;

fprintf('## Final running c = %g... \n',model_prior.c);
org_burn_in = sopts.burn_in;
sopts.burn_in = get_opt(sopts, 'burn_in_last', 10*org_burn_in);
sopts.init_W = get_opt(sopts, 'init_W', @(~,~,~,~)Ws{lmi}(:,:,end));
sopts.rho = rhos{lmi}{end};
rho_prior.log_alpha = log_alphas{lmi};
init_Wstats = ... 
  @(data,nfreq,rho_prior,model_prior)...
  W_stats_reg_model_ev( ...
    W,data{1},data{2},model_prior,get_opt(opts,'init_opts',struct));
[Ws, rhos] = ...
  bank({X(~tst_set,:), Y(~tst_set,:)}, nfreq, rho_prior, model_prior, update, init_Wstats, sopts);
W = Ws(:,:,end);
PhiW_all = [cos(X*W) sin(X*W)];
PhiW = PhiW_all(~tst_set,:);
PtP = PhiW'*PhiW;
PtY = PhiW'*Y(~tst_set,:);

if do_last_optimize
    beta = (PtP+model_prior.c*eye(2*nfreq))\(PtY);
    
    % use mode of posterior inv-gamma for noise variance
    if use_sigma2_error
        a_n = model_prior.a + (N-N_tst)/2.0;
        b_n = model_prior.b + 0.5* ( norm(Y(~tst_set)).^2 - PtY'*beta );
        sigma2_error = b_n/(a_n+1);
    else
        sigma2_error = 1;
    end
    fprintf('## BaNK {c=%g} sigma2_error: %g \n', model_prior.c, sigma2_error);
    
    z = rhos{end}.z;
    mus_z = rhos{end}.mus;
    mus_z = reshape(mus_z,size(mus_z,1),1,size(mus_z,2));
    mus_z = mus_z(:,:,z);
    Siginvs_z = rhos{end}.Sigmas;
    for l=1:size(Siginvs_z,3)
        Siginvs_z(:,:,l) = inv(Siginvs_z(:,:,l));
    end
    Siginvs_z = Siginvs_z(:,:,z);

%     lb_cell = {-inf(nfreq,1), -inf(nfreq,1), -inf(d,nfreq)};
%     ub_cell = { inf(nfreq,1),  inf(nfreq,1),  inf(d,nfreq)};
    auxdata = {X(~tst_set,:),Y(~tst_set,:),model_prior.c,mus_z,Siginvs_z,sigma2_error};
%     [~, ~, W] = ...
%         lbfgsb({beta(1:nfreq), beta(nfreq+1:end), W}, lb_cell, ub_cell,...
%                'rks_W_obj','rks_W_grad', auxdata,'print100callback',...
%                'm',25,'factr',1e-12, 'pgtol',1E-4,'maxiter',ceil(lbfgsb_maxiter/10)); %ceil(lbfgsb_maxiter/10)?
    optfunc = @(vars)rks_W_minfunc(vars,auxdata);
    opts_mf = struct;
    opts_mf.maxFunEvals = ceil(lbfgsb_maxiter/10);
    opts_mf.Display = 'off';
    opts_mf.Method = 'lbfgs';
    vars = minFunc(optfunc, [beta(1:nfreq); beta(nfreq+1:end); W(:)], opts_mf);
    W = reshape(vars(2*nfreq+1:2*nfreq+d*nfreq),d,nfreq);

    PhiW_all = [cos(X*W) sin(X*W)];
    PhiW = PhiW_all(~tst_set,:);
end

beta = (PhiW'*PhiW+lambdars(li)*eye(2*nfreq))\(PhiW'*Y(~tst_set,:));
BaNK_stats.Y_pred = PhiW_all(tst_set,:)*beta;
BaNK_stats.resids = BaNK_stats.Y_pred - Y(tst_set,:);
BaNK_stats.tst_err = mean(mean((Y(tst_set,:)-BaNK_stats.Y_pred).^2));
BaNK_stats.hol_mses = hol_mses;
BaNK_stats.lambdars = lambdars;
BaNK_stats.lambda_mult = lambda_mult;

BaNK_stats.Ws = Ws;
BaNK_stats.rhos = rhos;
BaNK_stats.beta = beta;
BaNK_stats.W = W;
BaNK_stats.priors = {rho_prior, model_prior};
BaNK_stats.time = toc(bank_stime);
fprintf('## BaNK Test MSE: %g \n', BaNK_stats.tst_err);

end