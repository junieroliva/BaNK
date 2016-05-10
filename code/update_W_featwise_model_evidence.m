function Wstats = ...
  update_W_featwise_model_evidence(Wstats,rho,data,model_prior,log_mod_ev_ratio,varargin)
% update_W_featwise_model_evidence This function samples random frequencies
%   in a round-robin MH fasion conditioned on the GMM spectral density and 
%   possibly other statistics. For each proposed new frequency w_j, we
%   calculate a log ratio of the model evidence w.r.t. the new proposed
%   frequency and the current frequency (using log_model_ev_ratio, see 
%   below); if the log ratio is greater than 0 or greater than log(rand) we
%   accept w_j, else reject. This is done in round-robin fashion (in random
%   order) for all frequencies.
% Inputs:
%   Wstats - struct of info for random frequencies and model with atleast  
%            the following feilds
%     .W - random frequency matrix
%     .PhiW (optional) - random feature matrix
%   rho - struct of GMM spectral density stats with feilds
%     .mus - d x k matrix of means for GMM
%     .Sigmas - d x d x k array of covariance matrices
%     .z - vector of the component assignments for each random frequency
%   data - data to use in model; e.g. for regression a cell with 
%          data{1} = X, the N x d matrix of input features, and
%          data{2} = Y, N x d2  matrix of output responses.
%   model_prior -  struct with model prior parameters and options
%   log_mod_ev_ratio - function handle
%     function [lme_ratio, Wstats_prop, Wstats, cos_plus, sin_plus] = ...
%     log_mod_ev_ratio(j,w_j_prop,Wstats,data,model_prior)
%        that returns the log ratio of model evidences (or an
%        approximation) for proposed frequency w_j_prop at index j w.r.t. a
%        current frequencies given in Wstats, using data and model_prior.
%        Returns: lme_ratio, the log ratio of model evidences; Wstats_prop,
%        stats using proposed frequency w_j_prop; Wstats, the stats using
%        current features (this is returned incase some stats were outdated
%        and were updated in the function call); cos_plus and sin_plus,
%        cosine and sin random features respectively.
%      Note, the log_mod_ev_ratio function *should not* return Wstats_prop  
%      with a feild of Wstats_prop.PhiW; Wstats_prop.PhiW is updated in 
%      this function only if w_j_prop is accepted in order to save on the 
%      memory overhead from unneccesarily copying PhiW.
% Output:
%   Wstats - struct of random frequencies (and stats, as above) after
%            sampling step

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
last_proc = get_opt(opts, 'last_proc', @(Wstats,~,~)Wstats);

W = Wstats.W;
nfreq = size(W,2);
update_PhiW = isfield(Wstats,'PhiW');

z = rho.z;
mus = rho.mus;
Sigmas = rho.Sigmas;
n_samp_MH = 1; % TODO: make an option?
jrprm = randperm(nfreq); 
accepted = 0;
for jj = 1:nfreq
    j = jrprm(jj);
    % sample Ws with MH step  
    for t = 1:n_samp_MH
        z_j_prop = z(j);

        % TODO: cache sqrt of Sigma matrices
        w_j_prop = mvnrnd(mus(:,z_j_prop)',Sigmas(:,:,z_j_prop),1)';
        [lme_ratio, Wstats_prop, Wstats, cos_plus, sin_plus] = ...
            log_mod_ev_ratio(j,w_j_prop,Wstats,data,model_prior);

        if lme_ratio>0 || log(rand)<=lme_ratio
            if update_PhiW
                PhiW_plus = Wstats.PhiW;
                Wstats = Wstats_prop;
                Wstats.PhiW = PhiW_plus;
                clear PhiW_plus; % should allow us to change PhiW in place
                Wstats.PhiW(:,[j j+nfreq]) = [cos_plus sin_plus];
            else
                Wstats = Wstats_prop;
            end
            % TODO: keep track of # of accepted here
            accepted = accepted + 1;
        end
    end
end
Wstats = last_proc(Wstats,data,model_prior);
Wstats.accepted = accepted;

end