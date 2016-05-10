function W_stats = W_stats_reg_model_ev(W,X,Y,model_prior,varargin)
% W_stats_reg_model_ev Get stats used in sampling W.
% Inputs:
%   W - d x nfreq matrix of random frequencies
%   X - N x d matrix of input covariates
%   Y - N x d2 matrix of output responses
%   model_prior -  struct with the follow feilds
%     (.a,.b) - inverse-gamma prior parameters to noise variance in linear
%             model
%     .c - scalar multiplier on identity matrix for beta weights
% Outputs:
%   W_stats - struct with the following feilds
%     .N - number of instances
%     .p - number of dimensions for responses
%     .W - random frequency matrix
%     .PhiW - matrix of random features for instances
%     .R - chol of Phi'*Pho+c*I, where c is prior parameter for beta (see
%         https://en.wikipedia.org/wiki/Bayesian_linear_regression#Other_cases)
%     .PtY - inner product of Phi'*Y (where Phi is matrix of random
%           features)
%     .YtY - norm of response vector
%     .betaW - betaW = PtP_cI\PtY, the mode for beta regression weights
%     .log_det_PtP_cI - log determinant of Phi'*Pho+c*I, where c is model
%                      prior parameter (see below).
%     .log_f - Get the log evidence of the model evidence w.r.t. random 
%             features.

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
save_PhiW = get_opt(opts, 'save_PhiW', true);

XW = X*W;
PhiW = [cos(XW) sin(XW)];

c = model_prior.c;

PtP_cI = PhiW'*PhiW+c*speye(size(PhiW,2));
R = chol(PtP_cI);
PtY = PhiW'*Y;

betaW = R\(R'\PtY); % betaW = PtP_cI\PtY;
log_det_PtP_cI = sum(2*log(diag(R))); % log_det_PtP_cI = log(det(PtP_cI));

W_stats.N = numel(Y);
W_stats.p = size(Y,2);
W_stats.W = W;
if save_PhiW
    W_stats.PhiW = PhiW;
end
W_stats.R = R;
W_stats.PtY = PtY;
W_stats.YtY = sum(Y(:).^2);
W_stats.betaW = betaW;
W_stats.PhiWbeta = PhiW*betaW;
W_stats.log_det_PtP_cI = log_det_PtP_cI;
W_stats.log_f = get_log_model_evidence_regression(W_stats,model_prior);
end