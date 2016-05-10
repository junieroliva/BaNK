function Wstats = laplace_bin_class(Wstats,data,model_prior,varargin)

% options
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
nbetas = get_opt(opts, 'nbetas', 1000);
sampMCMC = get_opt(opts, 'sampMCMC', false);
sample = get_opt(opts, 'sample', false);
dodebug = false;

X = data{1};
Y = data{2};
if length(data)>2
    bagsizes = data{3};
    lastinds = data{4};
else
    bagsizes = [];
end
do_mmd = iscell(X) | ~isempty(bagsizes);
W = Wstats.W;
N = size(Y,1);
nfreq = size(W,2);
beta_curr = get_opt(Wstats, 'mode', zeros(2*nfreq+1,1));
PhiW = get_opt(Wstats, 'PhiW');
if isempty(PhiW)
    if ~do_mmd
        XW = X*W;
        PhiW = [cos(XW) sin(XW) ones(N,1)];
    else
        if iscell(X)
            PhiW = cell2mat( cellfun(@(C)[mean([cos(C*W) sin(C*W)],1) 1], X, 'unif', false) );
        else
            PhiW = X*W; 
            PhiW = cumsum([cos(PhiW) sin(PhiW)],1);
            PhiW = PhiW(lastinds,:);
            PhiW(2:end,:) = PhiW(2:end,:) - PhiW(1:end-1,:);
            PhiW = bsxfun(@times,PhiW,1./bagsizes);
            PhiW = [PhiW ones(size(PhiW,1),1)];
        end
    end
    clear XW;
    if get_opt(opts, 'save_PhiW',false);
        Wstats.PhiW = PhiW;
    end
end
c = model_prior.c;

% run Newton's to find mode/precision for Laplace approx
[mode, precision] = newtons_binlr(PhiW,Y,c,beta_curr,opts);
% the variance is negative of the hessian
% S_inv = c*eye(2*nfreq+1);
% sig_predict = 1./(1+exp(-PhiW*beta_curr)); 
% precision = S_inv + PhiW'*bsxfun(@times,PhiW,sig_predict.*(1-sig_predict));

Wstats.mode = mode;
Wstats.precision = precision;
Wstats.PhiWbeta = PhiW*mode;

if sample
    if sampMCMC
        % sample from posterior with MCMC
        betas = MCMC_bin_posterior(PhiW,Y,mode,precision,c,nbetas,opts);
    else
        % sample from Laplace approx
        R = chol(precision);
        rndnm =  randn(length(mode),nbetas);
        betas = bsxfun(@plus,R\rndnm,mode);
        % get stats for sampled betas
        beta_mode = bsxfun(@minus,betas,mode);
        logpbetas = -.5*sum(beta_mode.*(precision*beta_mode),1);
        Wstats.logpbetas = logpbetas;
    end
    Wstats.betas = betas;
    Wstats.PhiWbetas = PhiW*betas;
else
    R = chol(precision);
    Wstats.logsqrtdetprec = sum(log(diag(R)));
    Wstats.laplace_mod_ev = Y'*Wstats.PhiWbeta - sum(log(1+exp(Wstats.PhiWbeta)),1) ... 
      - .5*model_prior.c*sum(Wstats.mode.^2) +  Wstats.logsqrtdetprec;
end



if dodebug && length(mode)==2 && rand<=.0005
    [bgrid1, bgrid2] = meshgrid(linspace(-.5,.5,100),linspace(-.5,.5,100));
    bgrid1 = bgrid1 + mode(1);
    bgrid2 = bgrid2 + mode(2);
    
    betas_grid = [bgrid1(:)';bgrid2(:)'];
    offset = opts.offset;
    logposts = Y'*bsxfun(@plus,PhiW*betas_grid,offset) ...
        -sum(log(1+exp(bsxfun(@plus,PhiW*betas_grid,offset))),1) -.5*c*sum(betas_grid.^2,1);
    beta_mode = bsxfun(@minus,betas_grid,mode);
    logposts = logposts - max(logposts);
    loglaplace = -.5*sum(beta_mode.*(precision*beta_mode),1);
    loglaplace = loglaplace - max(loglaplace);
    figure;
    subplot(2,2,1);
    contourf(bgrid1,bgrid2,reshape(exp(logposts),size(bgrid1)),100,'LineStyle','none');
    subplot(2,2,2);
    contourf(bgrid1,bgrid2,reshape(exp(loglaplace),size(bgrid1)),100,'LineStyle','none');
    
    subplot(2,2,3);
    contourf(bgrid1,bgrid2,reshape(exp(logposts)./exp(loglaplace),size(bgrid1)),100,'LineStyle','none');
    subplot(2,2,4);
    contourf(bgrid1,bgrid2,reshape(exp(logposts)-exp(loglaplace),size(bgrid1)),100,'LineStyle','none');
    mode
end
end