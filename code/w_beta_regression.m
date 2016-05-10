function [lme_ratio, Wstats_plus, Wstats, cos_plus, sin_plus] = ...
  w_beta_regression(j,w_j_plus,Wstats,data,model_prior,varargin)
% w_beta_regression Sample beta_j, w_j simultaneously given all other 
% parameters. We assume w_j_plus is sampled from it's marginal (given GMM
% parameters). We sample beta_j given w_j (and other parameters) by
% marginalizing sigma^2, the variance of response which has a inverse gamma
% distribution; resulting in a multivariate t distribution. The resulting 
% acceptance ratio is that of the model evidence for a 2d linear model
% w.r.t. the residual of a linear model in the set of [1:j-1 j+1:nfreq] 
% random frequencies.
% Inputs -
%   j - index of the random feature being sampled
%   w_j_plus - d x 1 vector for the proposed jth random frequency; assumed 
%              to have been drawn from a designated GMM component
%   Wstats - struct of stats associated with the current random frequencies
%            with atleast the following fields:
%     .W - d x nfreq matrix of current random frequencies
%     .betaW - 2*nfreq x 1 vector of linear weights on random features
%     .PhiWbeta - N x 1 vector of current predictions ( Phi_W(X)*betaW )
% Outputs - 
%   lme_ratio - log acceptance ratio for proposed w_j, beta_j
%   Wstats_plus - updated Wstats given proposed w_j, beta_j
%   Wstats - same as struct passed in (returned for interface w/ funcs)
%   cos_plus, sin_plus - N x 1 vector of cos(X*w_j_plus), sin(X*w_j_plus)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

X = data{1};
Y = data{2};
N = size(Y,1);

a = model_prior.a;
b = model_prior.b;
c = model_prior.c;

beta = Wstats.betaW;
nfreq = size(Wstats.W,2);

Xw_j_plus = X*w_j_plus;
Xw_j = X*Wstats.W(:,j);
cos_plus = cos(Xw_j_plus);
cos_curr = cos(Xw_j);
sin_plus = sin(Xw_j_plus);
sin_curr = sin(Xw_j);

PhiWj_curr = [cos_curr sin_curr];
delbetas = PhiWj_curr*beta([j j+nfreq]);
offsetj = Wstats.PhiWbeta-delbetas;
rj = Y - offsetj;
normrj2 = norm(rj)^2;
aj = a+N/2;

OmegaJ_curr = PhiWj_curr'*PhiWj_curr+c*eye(2);
betahatj_curr = OmegaJ_curr\(PhiWj_curr'*rj);
bj_curr = b + .5*(normrj2 - dot(rj,PhiWj_curr*betahatj_curr));
precj_curr = (aj/bj_curr).*OmegaJ_curr;
chol_precj_curr = chol(precj_curr);

PhiWj = [cos_plus sin_plus];
OmegaJ = PhiWj'*PhiWj+c*eye(2);
betahatj = OmegaJ\(PhiWj'*rj);
bj = b + .5*(normrj2 - dot(rj,PhiWj*betahatj));
precj = (aj/bj).*OmegaJ;
chol_precj = chol(precj);

%betaj = betahatj+mvtrnd(inv(precj),2*aj)'; % TODO: change/cache things?
betaj = betahatj + sqrt(2*aj/chi2rnd(2*aj))*(chol_precj\randn(2,1));

Wstats_plus = Wstats;
Wstats_plus.W(:,j) = w_j_plus;
Wstats_plus.betaW([j j+nfreq]) = betaj;
Wstats_plus.PhiWbeta = PhiWj*betaj + offsetj;

nume = -sum(log(diag(chol_precj)))-aj*log(bj); 
deno = -sum(log(diag(chol_precj_curr)))-aj*log(bj_curr); 
lme_ratio = nume - deno;

end