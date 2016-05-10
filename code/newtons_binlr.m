function [beta, H, obj] = newtons_binlr(PhiW,Y,c,beta_curr,varargin)
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

pI = c*eye(length(beta_curr));
offset = get_opt(opts, 'offset', 0);
[beta, obj] = newtons(beta_curr, {@get_log_f, @get_grad, @get_H}, opts);
H = get_H(beta);

    function log_f = get_log_f(beta)      
        PhiWbeta = PhiW*beta +offset;
        log_f = -Y'*PhiWbeta +sum(log(1+exp(PhiWbeta)),1) + .5*c*sum(beta.^2);
    end

    function grad = get_grad(beta)      
        sig_predict = 1./(1+exp(-(PhiW*beta+offset)));
        gradpen = c*beta;
        grad =  PhiW'*(sig_predict-Y) + gradpen;
    end

    function H = get_H(beta)  
        sig_predict = 1./(1+exp(-(PhiW*beta+offset)));
        H = PhiW' * bsxfun(@times,PhiW,sig_predict.*(1-sig_predict)) + pI;
    end
end