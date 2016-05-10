function lr_stats = binlogreg(X,Y,varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

if ~iscell(X)
    [N,d] = size(X);
else
    N = length(X);
    d = size(X{1},2);
end
Y = double(Y==1);

trn_set = get_opt(opts, 'trn_set');
hol_set = get_opt(opts, 'hol_set');
tst_set = get_opt(opts, 'tst_set');
if isempty(trn_set) || isempty(hol_set) || isempty(tst_set)
    tprec = get_opt(opts, 'tprec', .1);
    hprec = get_opt(opts, 'tprec', .1);
    [trn_set, hol_set, tst_set] = split_data( N, tprec, hprec );
end

sigma2s = get_opt(opts,'sigma2s');
if isempty(sigma2s)
    N_rot = get_opt(opts, 'N_rot', min(2000,N));
    pd2s = dists2(X(randperm(N,N_rot),:),X(randperm(N,N_rot),:));
    sigma2s = quantile(pd2s(:), .1:.2:.9);
end
lambdas = get_opt(opts,'lambdas',2.^(10:-1:-5));

nfreq = get_opt(opts,'nfreq',250);
W = get_opt(opts,'W',randn(d,nfreq));
bagsizes = get_opt(opts, 'bagsizes');
do_mmd = iscell(X) | ~isempty(bagsizes);
if ~isempty(bagsizes)
    lastinds = cumsum(bagsizes);
end
if ~do_mmd
    XW = X*W;
end

nsigma2s = length(sigma2s);
nlambdas = length(lambdas);
bopts = get_opt(opts, 'bopts', struct);
hol_err = nan(nsigma2s,nlambdas);
betas = cell(nsigma2s,nlambdas);
for si=1:nsigma2s
    sigma2 = sigma2s(si);
    if ~do_mmd
        PhiW = [cos(XW./sqrt(sigma2)) sin(XW./sqrt(sigma2)) ones(N,1)];
    else
        if iscell(X)
            PhiW = cell2mat( cellfun(@(C)[mean([cos(C*W./sqrt(sigma2)) sin(C*W./sqrt(sigma2))],1) 1], X, 'unif', false) );
        else
            PhiW = X*W./sqrt(sigma2); 
            PhiW = cumsum([cos(PhiW) sin(PhiW)],1);
            PhiW = PhiW(lastinds,:);
            PhiW(2:end,:) = PhiW(2:end,:) - PhiW(1:end-1,:);
            PhiW = bsxfun(@times,PhiW,1./bagsizes);
            PhiW = [PhiW ones(size(PhiW,1),1)];
        end
    end
    
    beta_si = zeros(size(PhiW,2),1);
    for li=1:nlambdas
        beta_si =  newtons_binlr(PhiW(~trn_set,:),Y(~trn_set),lambdas(li),beta_si,bopts);
        Yhol_pred = PhiW(hol_set,:)*beta_si>=0;
        hol_err(si,li) = sum(Y(hol_set)~=Yhol_pred)/length(Yhol_pred);
        betas{si,li} = beta_si;
    end
end

[si,li] = find(hol_err==min(hol_err(:)));
ri = randi(length(si));
si = si(ri); 
li = li(ri); 
beta_si = betas{si,li};
sigma2 = sigma2s(si);
if ~do_mmd
    PhiW = [cos(XW./sqrt(sigma2)) sin(XW./sqrt(sigma2)) ones(N,1)];
else
    if iscell(X)
        PhiW = cell2mat( cellfun(@(C)[mean([cos(C*W./sqrt(sigma2)) sin(C*W./sqrt(sigma2))],1) 1], X, 'unif', false) );
    else
        PhiW = X*W./sqrt(sigma2); 
        PhiW = cumsum([cos(PhiW) sin(PhiW)],1);
        PhiW = PhiW(lastinds,:);
        PhiW = PhiW - [zeros(1,size(PhiW,2)); PhiW(1:end-1,:)];
        PhiW = bsxfun(@times,PhiW,1./bagsizes);
        PhiW = [PhiW ones(size(PhiW,1),1)];
    end
end

beta = newtons_binlr(PhiW(~tst_set,:),Y(~tst_set),lambdas(li),beta_si,bopts);
Ytst_pred = PhiW(tst_set,:)*beta>=0;
tst_err = sum(Ytst_pred~=Y(tst_set))/length(Ytst_pred);

lr_stats.Y_pred = Ytst_pred;
lr_stats.resids = (Ytst_pred ~= Y(tst_set));
lr_stats.tst_err = tst_err;
lr_stats.beta = beta;
lr_stats.W = W;
lr_stats.sigma2 = sigma2;
lr_stats.lambda = lambdas(li);
lr_stats.hol_err = hol_err;
lr_stats.D = nfreq;
lr_stats.lambdars = lambdas;
end