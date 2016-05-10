function rho = update_rho_no_shared_cov(rho,W,rho_prior)
% update_rho_no_shared_cov sample spectral GMM given  
% Inputs:
%   rho - struct with the fields 
%     z (optional) - nfreq length vector of component labels for random 
%                    frequencies {default: ones vector}
%   W - d x nfreq matrix of random frequencies
%   rho_prior - struct with the fields (TODO: Avi explain these)
%     kappa - 
%     mu - 
%     nu - 
%     Psi - 
%     log_alpha (optional) - log probability of adding new component
%                            unnormalized (TODO: Avi check this)
%     la_quant (optional) - parameter to choose log_alpha if not given
%     K - number of components (or max number of components)
% Outputs:
%   rho - struct with fields
%     z - nfreq length vector of component labels for random 
%         frequencies 
%     mus - d x K component means
%     Sigmas - d x d x K component covariances
%     log_alpha - log probability of adding new component

[d,nfreq] = size(W);

log_alpha = get_opt(rho_prior,'log_alpha');
la_quant = get_opt(rho_prior,'la_quant',.01);
log_alpha = get_opt(rho,'log_alpha',log_alpha);
K = get_opt(rho_prior,'K',nfreq);
is_diagonal = get_opt(rho_prior,'is_diagonal',false);

z = get_opt(rho,'z',ones(nfreq,1));
count_comp = zeros(K,1);
Sigmas = nan(d,d,K);
if is_diagonal
    precision = nan(d,K);
end
mus = nan(d,K);
for l=1:K
    zl = z==l;
    prior_stat = get_prior_stat(W(:,zl),rho_prior);
    count_comp(l) = sum(zl);
    % sample rand feature distribution
    if ~is_diagonal 
        Sigmas(:,:,l) = iwishrnd(prior_stat.Psi,prior_stat.nu);        
    else
        precision(:,l) = gamrnd(prior_stat.gamma_alpha,1./prior_stat.gamma_beta);
        Sigmas(:,:,l) = diag(ones(d,1)./precision(:,l));
    end
    mus(:,l) = mvnrnd(prior_stat.mu',(1./prior_stat.kappa)*Sigmas(:,:,l))';
end

%sample the mixture component assignment
log_mvn_ps = zeros(K,nfreq);
for l=1:K
    if ~is_diagonal 
        SR = chol(Sigmas(:,:,l));
        log_mvn_ps(l,:) = -.5*sum((SR'\bsxfun(@minus,W,mus(:,l))).^2,1)'...
                      -.5*sum(2*log(diag(SR))) -(d/2)*log(2*pi);
    else
        D = bsxfun(@minus,W,mus(:,l));
        log_mvn_ps(l,:) = -0.5*sum(bsxfun(@times,D,precision(:,l)).*D, 1);
        log_mvn_ps(l,:) = log_mvn_ps(l,:) - 0.5*sum(log(diag(Sigmas(:,:,l)))) ;
    end
end
log_mvn_ps = bsxfun(@minus,log_mvn_ps,max(log_mvn_ps,[],1));
if(isempty(log_alpha)) 
    % get ROT log_alpha since it's hard to judge the likelihoods ahead 
    % of time
    log_alpha = quantile( abs(log_mvn_ps(log_mvn_ps<0)), la_quant );
    fprintf('{log_alpha: %g}\n',log_alpha);
end
for j =1:nfreq
    count_comp(z(j)) = count_comp(z(j))-1;
    log_count_comp = log(count_comp);
    new_cls = find(count_comp==0,1);
    if (~isempty(new_cls))
        log_count_comp(new_cls) = log_alpha;
    end
    log_pdf = log_mvn_ps(:,j) + log_count_comp;
    cdf = cumsum( exp(log_pdf-max(log_pdf)) );
    z(j) = find(rand*cdf(end)<cdf,1);
    count_comp(z(j)) = count_comp(z(j))+1;
end

% save rho
rho.z = z;
rho.mus = mus;
rho.Sigmas = Sigmas;
rho.log_alpha = log_alpha;

    function prior_stat = get_prior_stat(W,rho_prior)
        kappa = rho_prior.kappa;
        mu = rho_prior.mu;
        kappa_mu = mu*kappa;
        if ~is_diagonal
            nu = rho_prior.nu;
            Psi = rho_prior.Psi;
        else
            gamma_alpha = rho_prior.gamma_alpha;
            gamma_beta = rho_prior.gamma_beta;
        end
        
    
        n = size(W,2);
        if n>0
            xbar = mean(W,2);
            C = bsxfun(@minus,W,xbar);
            if ~is_diagonal
                C = C*C';
            end
        else
            xbar = zeros(d,1);
            C = zeros(d,d);
        end

        prior_stat.mu = (kappa_mu + n*xbar)/(kappa+n);
        prior_stat.kappa = kappa+n;
        if ~is_diagonal
            prior_stat.nu = nu + n;
            prior_stat.Psi = Psi + C + (kappa*n /(kappa+n))*(xbar-mu)*(xbar-mu)';
        else
            prior_stat.gamma_alpha = gamma_alpha + n;
            prior_stat.gamma_beta = gamma_beta + 0.5*sum(C.*C,2)+ (kappa*n /(kappa+n))*(xbar-mu).*(xbar-mu);
        end
    end

end