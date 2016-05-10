function log_f = get_log_model_evidence_regression(W_stats,model_prior)
% get_log_model_evidence_regression Get the log evidence of the model
% evidence w.r.t. random features.
% Inputs:
%   W_stats - struct with the following feilds
%     .N - number of instances
%     .p - number of dimensions for responses
%     .YtY - norm of response vector
%     .PtY - inner product of Phi'*Y (where Phi is matrix of random
%           features)
%     .log_det_PtP_cI - log determinant of Phi'*Pho+c*I, where c is model
%                      prior parameter (see below).
%     .betaW - betaW = PtP_cI\PtY, the mode for beta regression weights
%   model_prior - struct with the follow feilds
%     (.a,.b) - inverse-gamma prior parameters to noise variance in linear
%             model
% Outputs:
%   log_f - log model evidence up to constants

a = model_prior.a;
b = model_prior.b;

N = W_stats.N;
p = W_stats.p;

log_f= -.5*W_stats.log_det_PtP_cI^p ...
    -(a+N/2)*log(b+.5*(W_stats.YtY-dot(W_stats.PtY(:),W_stats.betaW(:))));
end