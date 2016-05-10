function [obj, grad] = rks_W_vec(vars, auxdata)
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

obj = rks_W_obj(beta_cos, beta_sin, W, auxdata);
[dbeta_cos, dbeta_sin, dW] = rks_W_grad(beta_cos, beta_sin, W, auxdata);
grad = [dbeta_cos(:); dbeta_sin(:); dW(:)];

end