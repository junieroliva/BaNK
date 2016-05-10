function Wstats = initWstats_multi_class( W, data ,model_prior )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Wstats.W = W;

X = data{1};
Y = data{2};
if length(data)>2
    bagsizes = data{3};
    lastinds = data{4};
else
    bagsizes = [];
end
do_mmd = iscell(X) | ~isempty(bagsizes);
W = Wstats.W;
[N,C] = size(Y);
nfreq = size(W,2);
beta_curr = zeros(2*nfreq+1,C);

if ~do_mmd
    XW = X*W;
    PhiW = [cos(XW) sin(XW) ones(N,1)];
else
    if iscell(X)
        PhiW = cell2mat( cellfun(@(C)[mean([cos(C*W) sin(C*W)],1) 1], X, 'unif', false) );
    else
        PhiW = X*W; 
        PhiW = cumsum([cos(PhiW) sin(PhiW)],1);
        PhiW = PhiW(lastinds,:);
        PhiW(2:end,:) = PhiW(2:end,:) - PhiW(1:end-1,:);
        PhiW = bsxfun(@times,PhiW,1./bagsizes);
        PhiW = [PhiW ones(size(PhiW,1),1)];
    end
end
clear XW;

Wstats.mode = quasinewtons_multilr(PhiW,Y,model_prior.c,beta_curr);
beta = reshape(Wstats.mode, [], C);
Wstats.PhiWbeta = PhiW*beta;

end

