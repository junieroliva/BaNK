function [lme_ratio, Wstats_plus, Wstats, cos_plus, sin_plus] = ...
  laplace_mod_ev_bin_class(j,w_j_plus,Wstats,data,model_prior,varargin)

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
opts.sample = false;

X = data{1};

nfreq = size(Wstats.W,2);
Xw_j_plus = X*w_j_plus;
cos_plus = cos(Xw_j_plus);
sin_plus = sin(Xw_j_plus);

Wstats_plus = Wstats;
Wstats_plus.W(:,j) = w_j_plus;
Wstats_plus.PhiW(:,[j j+nfreq]) = [cos_plus sin_plus];
Wstats_plus = laplace_bin_class(Wstats_plus,data,model_prior,opts);

lme_ratio = Wstats_plus.laplace_mod_ev - Wstats.laplace_mod_ev;


end