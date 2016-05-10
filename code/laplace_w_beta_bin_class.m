function [lme_ratio, Wstats_plus, Wstats, cos_plus, sin_plus] = ...
  laplace_w_beta_bin_class(j,w_j_plus,Wstats,data,model_prior,varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
opts.sample = false;

X = data{1};
Y = data{2};
if length(data)>2
    bagsizes = data{3};
    lastinds = data{4};
else
    bagsizes = [];
end
do_mmd = iscell(X) | ~isempty(bagsizes);
N = size(Y,1);

c = model_prior.c;

beta = Wstats.mode;
nfreq = size(Wstats.W,2);

if ~do_mmd
    Xw_j_plus = X*w_j_plus;
    Xw_j = X*Wstats.W(:,j);
    cos_plus = cos(Xw_j_plus);
    cos_curr = cos(Xw_j);
    sin_plus = sin(Xw_j_plus);
    sin_curr = sin(Xw_j);
else
    if iscell(X)
        cos_plus = cellfun(@(C)mean(cos(C*w_j_plus),1), X);
        sin_plus = cellfun(@(C)mean(sin(C*w_j_plus),1), X);
        cos_curr = cellfun(@(C)mean(cos(C*Wstats.W(:,j)),1), X);
        sin_curr = cellfun(@(C)mean(sin(C*Wstats.W(:,j)),1), X);
    else
        Xw_j_plus = X*w_j_plus; 
        cos_plus = cumsum(cos(Xw_j_plus),1);
        cos_plus = cos_plus(lastinds,:);
        cos_plus = cos_plus - [0; cos_plus(1:end-1,:)];
        cos_plus = bsxfun(@times,cos_plus,1./bagsizes);
        sin_plus = cumsum(sin(Xw_j_plus),1);
        sin_plus = sin_plus(lastinds,:);
        sin_plus = sin_plus - [0; sin_plus(1:end-1,:)];
        sin_plus = bsxfun(@times,sin_plus,1./bagsizes);
        clear Xw_j_plus;
        Xw_j = X*Wstats.W(:,j);
        cos_curr = cumsum(cos(Xw_j),1);
        cos_curr = cos_curr(lastinds,:);
        cos_curr = cos_curr - [0; cos_curr(1:end-1,:)];
        cos_curr = bsxfun(@times,cos_curr,1./bagsizes);
        sin_curr = cumsum(sin(Xw_j),1);
        sin_curr = sin_curr(lastinds,:);
        sin_curr = sin_curr - [0; sin_curr(1:end-1,:)];
        sin_curr = bsxfun(@times,sin_curr,1./bagsizes);
        clear Xw_j;
    end
end
opti_j.nbetas = 1;
opti_j.sampMCMC = false;
opti_j.sample = true;
if j~=nfreq
    delbetas = [cos_curr sin_curr]*beta([j j+nfreq]);
    opti_j.offset = Wstats.PhiWbeta -delbetas;
    Wstats_j.PhiW = [ cos_plus sin_plus ];
    Wstats_j.mode = beta([j j+nfreq]);
    Wstats_j_curr.PhiW = [ cos_curr sin_curr ];
    Wstats_j_curr.mode = beta([j j+nfreq]);
else
    delbetas = [cos_curr sin_curr ones(N,1)]*beta([j j+nfreq end]);
    opti_j.offset = Wstats.PhiWbeta -delbetas;
    Wstats_j.PhiW = [ cos_plus sin_plus ones(N,1) ];
    Wstats_j.mode = beta([j j+nfreq end]);
    Wstats_j_curr.PhiW = [ cos_curr sin_curr ones(N,1) ];
    Wstats_j_curr.mode = beta([j j+nfreq end]);
end
Wstats_j.W = w_j_plus;
Wstats_j_curr.W = Wstats.W(:,j);

Wstats_j = laplace_bin_class(Wstats_j,data,model_prior,opti_j);
Wstats_j_curr = laplace_bin_class(Wstats_j_curr,data,model_prior,opti_j);
%Wstats_j.betas = Wstats_j.mode; % HACK

Wstats_plus = Wstats;
Wstats_plus.W(:,j) = w_j_plus;
lin_plus = Wstats_j.PhiW*Wstats_j.betas +opti_j.offset;
prec_plus = Wstats_j.precision;
mode_plus = Wstats_j.mode;
prec_curr = Wstats_j_curr.precision;
mode_curr = Wstats_j_curr.mode;
if j~=nfreq
    nume = Y'*lin_plus -sum(log(1+exp(lin_plus)),1) -.5*c*sum(Wstats_j.betas.^2)...
           -.5*(beta([j j+nfreq])-mode_curr)'*prec_curr*(beta([j j+nfreq])-mode_curr)...
           +sum(log(diag(chol(prec_curr)))); %.5*log(det(precision))
    lin_curr = Wstats.PhiWbeta;
    deno = Y'*lin_curr -sum(log(1+exp(lin_curr)),1) -.5*c*sum(beta([j j+nfreq]).^2)...
           -.5*(Wstats_j.betas-mode_plus)'*prec_plus*(Wstats_j.betas-mode_plus)...
           +sum(log(diag(chol(prec_plus)))); %.5*log(det(prec_plus))
    Wstats_plus.mode([j j+nfreq]) =  Wstats_j.betas;
else
    nume = Y'*lin_plus -sum(log(1+exp(lin_plus)),1) -.5*c*sum(Wstats_j.betas.^2)...
           -.5*(beta([j j+nfreq end])-mode_curr)'*prec_curr*(beta([j j+nfreq end])-mode_curr)...
           +sum(log(diag(chol(prec_curr)))); %.5*log(det(precision))
    lin_curr = Wstats.PhiWbeta;
    deno = Y'*lin_curr -sum(log(1+exp(lin_curr)),1) -.5*c*sum(beta([j j+nfreq end]).^2)...
           -.5*(Wstats_j.betas-mode_plus)'*prec_plus*(Wstats_j.betas-mode_plus)...
           +sum(log(diag(chol(prec_plus)))); %.5*log(det(prec_plus))
    Wstats_plus.mode([j j+nfreq end]) =  Wstats_j.betas;
end
lme_ratio = nume - deno;

Wstats_plus.PhiWbeta = lin_plus;

end