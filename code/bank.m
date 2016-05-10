function [Ws,rhos] = ...
    bank(data,nfreq,rho_prior,model_prior,update,init_Wstats,varargin)
% BaNK function to run the BaNK framework.
% Inputs: 
%   data - Input data to work over. E.g. can be a cell array {X Y} for
%          supervised problems, or a matrix X for unsupervized problems.
%   nfreq - Number of random features to use.
%   rho_prior - Prior parameters for the distribution of random features.
%               (Can also include options for update functions below.)
%   model_prior - Prior parameters for the distribution of model specific
%                 parameters. E.g. normal parameters mu and Sigma for
%                 linear regression. (Can also include options for update
%                 functions below.)
%   update - struct with the following function handles
%     function rho = update.rho(rho,W,rho_prior):
%       Function that samples the posterior of rho given W, rest. E.g.
%       samples mean, covariances for a GMM given W and component
%       assignments.
%     function Wstats = update.W(Wstats,rho,data,model_prior)
%       Function that samples random frequencies, and other model stats
%       (like beta weights if not marginalized out) given rho (and posibly
%       component assignments in Z). Note, this function should handle the
%       case where Wstats is given with only Wstats.W as a feild, which 
%       will occur in the first iteration.
%   init_Wstats - function that returns initial random frequencies, and 
%                 other model stats:
%       function Wstats = init_Wstats(data,nfreq,rho_prior,model_prior) 
%   opts (optional) - struct with optional parameters with {defualts} 
%     burn_in - number of draws to burn in {10}
%     samp_gap - number of draws to discard between samples {100}
%     nsamps - number of samples to return {1}
%     itprint - function itprint(i,data,rho,Wstats) that
%       prints to stdout (or does any other sort of logging/processing) 
%       on the iterates that save a sample {@(~,~,~,~,~)[]}.
% Outputs:
%   Ws - 3d array of random frequency samples where the last dimension is
%        length nsamp and Ws(:,:,i) is the ith matrix of random
%        frequencies.
%   rhos - cell array of length nsamp, where rhos{i} is the ith random
%          frequency distribution drawn from posterior.

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

% sampling options
burn_in = get_opt(opts, 'burn_in', 10);
samp_gap = get_opt(opts, 'samp_gap', 100);
n_samp = get_opt(opts, 'n_samp', 1);

% get initial random frequencies
Wstats = init_Wstats(data,nfreq,rho_prior,model_prior);

% printing function
itprint = get_opt(opts, 'itprint', @(~,~,~,~,~)[]);

% main loop
tot_iters = burn_in + samp_gap*n_samp;
Ws = nan(size(Wstats.W,1),nfreq,n_samp);
rhos = cell(n_samp,1);
rho = get_opt(opts, 'rho', struct);
for i=1:tot_iters
    % sample posterior for rho given W
    rho = update.rho(rho,Wstats.W,rho_prior);
    
    % sample posterior for W given rho
    Wstats = update.W(Wstats,rho,data,model_prior);
    
    % save?
    if i>burn_in && mod(i-burn_in,samp_gap)==0
        Ws(:,:,(i-burn_in)/samp_gap) = Wstats.W;
        rhos{(i-burn_in)/samp_gap} = rho;
    end
    
    itprint(i,data,rho,Wstats,rho_prior,model_prior);
end

end

