function obj = rks_W_obj(beta_cos, beta_sin, W, auxdata, varargin)
% beta_cos: D x 1
% beta_sin: D x 1
% W: d x D
% auxdata: {X,Y,c,mus_z,Siginvs_z}
%   X: N x d
%   Y: N x 1
%   mus_z: d x 1 x D
%   Siginvs_z: d x d x D
%   c: real > 0

if length(auxdata)==5 % use defualt
    [X,Y,c,mus_z,Siginvs_z] = deal(auxdata{:});
    sigma2_error = 1;
else
    [X,Y,c,mus_z,Siginvs_z,sigma2_error] = deal(auxdata{:});
end
[d,D] = size(W);
W3d = reshape(W,d,1,D);
W_mu = W3d-mus_z;
W_nll = .5*sum( mtimesx(W_mu,'t',mtimesx(Siginvs_z,W_mu)) );

beta_nll = .5*c*(sum(beta_cos(:).^2)+sum(beta_sin(:).^2));

XW = X*W;
preds = cos(XW)*beta_cos + sin(XW)*beta_sin;
r = Y-preds;
obj = .5*sum(r.^2)+beta_nll+W_nll*sigma2_error;

end