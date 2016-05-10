function [dbeta_cos, dbeta_sin, dW] = rks_W_grad(beta_cos, beta_sin, W, auxdata, varargin)
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
dW_nll = squeeze(mtimesx(Siginvs_z,W_mu));

XW = X*W;
cosXW = cos(XW);
sinXW = sin(XW);
preds = cosXW*beta_cos + sinXW*beta_sin;
Del = bsxfun(@times,sinXW,beta_cos') - bsxfun(@times,cosXW,beta_sin');
r = Y-preds;
XtR = bsxfun(@times,X',r');

dW = XtR*Del + dW_nll*sigma2_error;
dbeta_cos = -cosXW'*r + c*beta_cos;
dbeta_sin = -sinXW'*r + c*beta_sin;

end