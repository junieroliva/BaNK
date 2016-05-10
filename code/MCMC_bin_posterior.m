function betas = MCMC_bin_posterior(PhiW,Y,mode,precision,c,n_samp,varargin)
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
burn_in = get_opt(opts, 'burn_in', 100);
samp_gap = get_opt(opts, 'samp_gap', 100);
offset = get_opt(opts, 'offset', 0);
d = size(PhiW,2);

% sample from Laplace approx
L = chol(precision);
beta =  L\randn(d,1)+mode;
% get beta stats
PB = PhiW*beta+offset;
expnPB = exp(-PB);
normB2 = sum(beta(:).^2);
logpbeta = -.5*(beta-mode)'*precision*(beta-mode);
tot_iters = burn_in + samp_gap*n_samp;
betas = nan(d,n_samp);
accepted = 0;
for i=1:tot_iters
    % sample from Laplace approx
    beta_prop =  L\randn(d,1)+mode;
    % get proposal stats
    PB_prop = PhiW*beta_prop+offset;
    PDB = PB_prop-PB;
    normB2_prop = sum(beta_prop(:).^2);
    logpbeta_prop = -.5*(beta_prop-mode)'*precision*(beta_prop-mode);
    % get log acceptance ratio
    logratio = Y'*PDB -sum(log( (expnPB+exp(PDB))./(1+expnPB) ),1) ...
        -5*c*(normB2_prop-normB2);
    logratio = logratio +logpbeta -logpbeta_prop;
    
    % accept?
    if logratio>0 || log(rand)<=logratio
        beta = beta_prop;
        PB = PB_prop;
        expnPB = exp(-PB);
        normB2 = normB2_prop;
        logpbeta = logpbeta_prop;
        accepted = accepted +1;
    end
    % save?
    if i>burn_in && mod(i-burn_in,samp_gap)==0
        betas(:,(i-burn_in)/samp_gap) = beta;
    end
end


end