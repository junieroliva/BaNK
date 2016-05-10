function [W, pred, beta] = cdesc_rks_W( X,Y,W,beta,pred,...
                             c,mus_z,Siginvs_z,sigma2_error, varargin )

if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end
citers = get_opts(opts, 'citers', 25);
liters = get_opts(opts, 'liters', 100);

nfreq = size(W,2);
beta_cos = beta(1:nfreq);
beta_sin = beta(nfreq+1:2*nfreq);
lb_cell = {-inf(nfreq,1), -inf(nfreq,1), -inf(d,nfreq)};
ub_cell = { inf(nfreq,1),  inf(nfreq,1),  inf(d,nfreq)};
for ci=1:citers
    jprm = randperm(nfreq);
    for jj=1:nfreq
        j = jprm(jj);
        
        XWj = X*W(:,j);
        offset = pred - (cos(XWj)*beta_cos(j) + sin(XWj)*beta_sin(j));
        
        auxdata = {X,Y,c,mus_z(:,1,j),Siginvs_z(:,:,j),sigma2_error,offset};
        [beta_cos(j), beta_sin(j), W(:,j)] = ...
          lbfgsb( {beta_cos(j), beta_sin(j),  W(:,j)}, lb_cell, ...
            ub_cell, 'rks_W_obj','rks_W_grad', auxdata,'print100callback', ...
            'm',25,'factr',1e-12, 'pgtol',1E-4,'maxiter',liters );
        
        pred = offset + (cos(XWj)*beta_cos(j) + sin(XWj)*beta_sin(j));
    end
end

end