function funcs = newtons_multilr_funcs(PhiW,Y,c,beta_curr,varargin)
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

C = size(Y, 2);
p = size(PhiW, 2);
pI = c*eye(length(beta_curr));
offset = get_opt(opts, 'offset', 0);

vec_hess = get_opt(opts,'vec_hess',false);
do_test = get_opt(opts,'do_test',false);
errtol = get_opt(opts,'errtol',1E-10);

if vec_hess || do_test
    PhiWoprods = permute(PhiW',[1 3 2]);
    PhiWoprods = mtimesx(PhiWoprods,PhiWoprods,'T');
    PhiWoprods = reshape(PhiWoprods,p^2,N)';
    diaginds = repmat(eye(C)>0,1,1,N);
    diaginds = reshape(diaginds,C^2,N);
end

funcs = {@get_log_f, @get_grad, @get_H};

    function log_f = get_log_f(beta) 
        B = reshape(beta, C, p);
        PhiWB = bsxfun(@plus,PhiW*B',offset);

        M = min(PhiWB, [], 2);
        PhiWBt_s = bsxfun(@minus, PhiWB, M);

        log_f = -sum(sum(Y.*PhiWB)) + sum(log( sum(exp(PhiWBt_s),2) )) + sum(M) + 0.5*c*sum(beta.^2);
    end

    function grad = get_grad(beta)      
        B = reshape(beta, C, p);
        
        PhiWB = bsxfun(@plus,PhiW*B',offset);
        PhiWB = bsxfun(@minus, PhiWB, min(PhiWB,[],2));
        
        P = exp(PhiWB);
        P = bsxfun(@times, P, 1./sum(P,2));
        grad = vec((P - Y)'*PhiW) + c*beta;
    end

    function H = get_H(beta)  
        B = reshape(beta, C, p);
        PhiWB = bsxfun(@plus,PhiW*B',offset);

        M = min(PhiWB, [], 2);
        PhiWB = bsxfun(@minus, PhiWB, M);
        
        P = exp(PhiWB);
        P = bsxfun(@times, P, 1./sum(P,2));
        
        if vec_hess || do_test
            % outer products of class probabilities
            W = permute(P',[1 3 2]);
            W = mtimesx(W,W,'T');
            W = -reshape(W,C^2,N);
            W(diaginds) = reshape(P',[],1)+W(diaginds);
            W = W';

            % do kronecker product, sum instances
            H = bsxfun(@times, reshape(W,N,1,C^2), PhiWoprods);
            H = squeeze(sum(H,1));

            % reshape to kron dimensions
            H = reshape(H,p,p,C,C);
            H = reshape(permute(H,[ 1 3 2 4 ]), C*p,C*p);
        end
        
        % test vectorize Hessian vs with for loops
        if ~vec_hess || do_test
            H2 = nan(C*p,C*p);
            for j=1:C
                i1 = (j-1)*p+1:j*p;
                H2(i1,i1) = PhiW'*bsxfun(@times,P(:,j).*(1-P(:,j)),PhiW);
                for k=j+1:C
                    i1 = (j-1)*p+1:j*p;
                    i2 = (k-1)*p+1:k*p;
                    H2(i1,i2) = PhiW'*bsxfun(@times,-P(:,j).*P(:,k),PhiW);
                    H2(i2,i1) = H2(i1,i2)';
                end
            end
            if do_test 
                if max(abs(H(:)-H2(:)))>errtol
                    error('Hessians do not match');
                end
            else
                H = H2;
            end
            clear H2;
        end
        
        H = H + pI;
    end
end