function [lme_ratio, W_stats_plus, W_stats, cos_plus, sin_plus] = ...
    lowrank_update_W_stats_reggression(j,w_j_plus,W_stats,data,model_prior)
% lowrank_update_W_stats_reggression TODO: write this

very_verbose = false; % TODO: fix, make option?

X = data{1};
Y = data{2};
c = model_prior.c;

% Update PhiW, PhiW'*Y, and chol(PhiW'*PhiW+cI)
nfreq = size(W_stats.W,2);
PhiW = W_stats.PhiW;
Xw_j_plus = X*w_j_plus;

cos_plus = cos(Xw_j_plus);
cos_curr = PhiW(:,j);
cos_delta = cos_plus-cos_curr;
norm_cos_plus = norm(cos_plus);
norm_cos = norm(cos_curr);
norm_cos_delta = norm_cos_plus-norm_cos;

sin_plus = sin(Xw_j_plus);
sin_curr = PhiW(:,j+nfreq);
sin_delta = sin_plus-sin_curr;
norm_sin_plus = norm(sin_plus);
norm_sin = norm(sin_curr);
norm_sin_delta = norm_sin_plus-norm_sin;

if abs(norm_cos_delta)<=1E-5 || abs(norm_sin_delta)<=1E-5% numerical errors start creeping in here ?
    % I think it's a rank 3 update when norm_j_delta=0 ...
    if very_verbose
        warning('Non-rank-2 update.');
    end
    PhiW(:,j) = cos_plus;
    PhiW(:,j+nfreq) = sin_plus;
    PtP_cI = PhiW'*PhiW+c*speye(size(PhiW,2));
    R = chol(PtP_cI);
else
    % updates
    u = ([cos_delta sin_delta]'*PhiW)';

    % cosine updates
    %u1 = (cos_delta'*PhiW)'./norm_cos_delta;
    u1 = u(:,1)./norm_cos_delta;
    u1(j) = norm_cos_plus;
    v1 = u1;
    v1(j) = norm_cos;

    % sine updates
    %u2 = (sin_delta'*PhiW)'./norm_sin_delta;
    u2 = u(:,2)./norm_sin_delta;
    u2(j) = dot(sin_delta,cos_plus)./norm_sin_delta;
    u2(j+nfreq) = norm_sin_plus;
    v2 = u2;
    v2(j+nfreq) = norm_sin;

    % update chol of PtP_cI
    R = W_stats.R;
    R = cholupdate(R,u1,'+');
    R = cholupdate(R,u2,'+');
    try
        R = cholupdate(R,v1,'-'); 
        R = cholupdate(R,v2,'-'); 
    catch
        if very_verbose
            warning('Inaccurate Cholesky decomposition.');
        end
        W_stats = W_stats_reg_model_ev(W_stats.W,data{1},data{2},model_prior);
        PhiW(:,j) = cos_plus;
        PhiW(:,j+nfreq) = sin_plus;
        PtP_cI = PhiW'*PhiW+c*speye(size(PhiW,2));
        R = chol(PtP_cI);
    end
end

PtY = W_stats.PtY;
PtY(j,:) = cos_plus'*Y;
PtY(j+nfreq,:) = sin_plus'*Y;

% get rest of updated stats
betaW = R\(R'\PtY);
log_det_PtP_cI = sum(2*log(diag(R))); 

% save stats
W_stats_plus.N = numel(Y);
W_stats_plus.p = size(Y,2);
W_stats_plus.W = W_stats.W;
W_stats_plus.W(:,j) = w_j_plus;
%W_stats_plus.PhiW = PhiW; % TO SAVE TIME DON'T UPDATE UNLESS ACCEPTED
W_stats_plus.R = R;
W_stats_plus.PtY = PtY;
W_stats_plus.YtY = W_stats.YtY;
W_stats_plus.betaW = betaW;
W_stats_plus.log_det_PtP_cI = log_det_PtP_cI;
W_stats_plus.log_f = get_log_model_evidence_regression(W_stats_plus,model_prior);

lme_ratio = W_stats_plus.log_f - W_stats.log_f;

end