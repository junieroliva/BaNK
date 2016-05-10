function [obj, grad] = rks_W_minfunc(vars, auxdata)
% beta_cos: D x 1
% beta_sin: D x 1
% W: d x D
% auxdata: {X,Y,c,mus_z,Siginvs_z}
%   X: N x d
%   Y: N x 1
%   c: real > 0
%   mus_z: d x 1 x D
%   Siginvs_z: d x d x D


[d,~,D] = size(auxdata{5});

ind = 0;
beta_cos = vars(ind+1:ind+D);
ind = ind+D;
beta_sin = vars(ind+1:ind+D);
ind = ind+D;
W = reshape(vars(ind+1:ind+d*D),d,D);

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
dW_nll = squeeze(mtimesx(Siginvs_z,W_mu));

beta_nll = .5*c*(sum(beta_cos(:).^2)+sum(beta_sin(:).^2));

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
grad = [dbeta_cos(:); dbeta_sin(:); dW(:)];
obj = .5*sum(r.^2)+beta_nll+W_nll*sigma2_error;


end