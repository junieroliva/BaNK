function [lme_ratio, Wstats_plus, Wstats, cos_plus, sin_plus] = ...
  imp_samp_bin_class(j,w_j_plus,Wstats,data,model_prior,varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
sampMCMC = get_opt(opts, 'sampMCMC', true);

X = data{1};
Y = data{2};
betas = Wstats.betas;
PhiWbetas = Wstats.PhiWbetas;
c = model_prior.c;

nfreq = size(Wstats.W,2);
PhiW = Wstats.PhiW;
Xw_j_plus = X*w_j_plus;

cos_plus = cos(Xw_j_plus);
cos_curr = PhiW(:,j);
cos_delta = cos_plus-cos_curr;

sin_plus = sin(Xw_j_plus);
sin_curr = PhiW(:,j+nfreq);
sin_delta = sin_plus-sin_curr;

DelBetas = [cos_delta sin_delta]*betas([j j+nfreq],:);
YtDB = Y'*DelBetas;
expPB = exp(-PhiWbetas);
PhiWbetas_plus = PhiWbetas + DelBetas;

%ratio = exp( YtDB - sum(log( (expPB+exp(DelBetas))./(1+expPB) ),1) );
logratio = YtDB - sum(log( (expPB+exp(DelBetas))./(1+expPB) ),1);

% max(ratio)
% 
% PhiW2 = PhiW;
% PhiW2(:,j) = cos_plus;
% PhiW2(:,j+nfreq) = sin_plus;
% ratio2 = exp( Y'*(PhiW2-PhiW)*betas  ...
%               -sum(log((1+exp(PhiW2*betas))./(1+exp(PhiW*betas))),1) );
% max(abs(ratio-ratio2))
% YtDB2 = Y'*(PhiW2-PhiW)*betas;
% max(abs(YtDB-YtDB2))
% sum_log = sum(log( (expPB+exp(DelBetas))./(1+expPB) ),1);
% sum_log2 = sum(log((1+exp(PhiW2*betas))./(1+exp(PhiW*betas))),1);
% max(abs(sum_log-sum_log2))
% 
% pratio = prod( (exp(bsxfun(@times,Y,PhiW2*betas))./(1+exp(PhiW2*betas))) ./ ...
%                (exp(bsxfun(@times,Y,PhiW*betas)) ./(1+exp(PhiW*betas) )), 1);
% pratio2 = ratio;
% pratio2(isinf(ratio)) = max(ratio(~isinf(ratio)) );
%ratio(isinf(ratio)) = max(ratio(~isinf(ratio)) );

if sampMCMC
    ratio = mean(exp(logratio));
else
    logpbetas = Wstats.logpbetas;
    logweights = Y'*PhiWbetas_plus -sum(log(1+exp(PhiWbetas_plus)),1) -.5*c*sum(betas.^2,1) -logpbetas;
    logratioweights = logratio + logweights;
    normweights = sum( exp( logweights ) );
    % weights = exp( logweights );
    % weights = weights'./sum(weights);
    % [ratio*weights pratio2*weights]
    % ratio = ratio*weights;
    ratio = sum( exp(logratioweights)./normweights );
end
lme_ratio = log(ratio);

Wstats_plus = Wstats;
Wstats_plus.PhiWbetas = PhiWbetas_plus;
Wstats_plus.W(:,j) = w_j_plus;
%Wstats_plus = laplace_bin_class(Wstats_plus,data,model_prior);
end