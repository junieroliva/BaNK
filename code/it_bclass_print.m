function it_bclass_print(i,data,rho,Wstats,~,model_prior)
Y = data{2};

beta = Wstats.mode;
if isfield(Wstats,'PhiWbeta')
    PhiWbeta = Wstats.PhiWbeta;
else
    PhiWbeta = Wstats.PhiW*Wstats.betaW;
end

err = mean(Y ~= (PhiWbeta>=0));
log_f = -Y'*PhiWbeta +sum(log(1+exp(PhiWbeta)),1) + .5*model_prior.c*sum(beta.^2);

if ~isfield(Wstats,'accepted')
    fprintf('[i: %i]\t{err: %g}\t{nll: %g}\t{ncomps: %i}\n',...
        i,err,log_f,length(unique(rho.z)));
else
    fprintf('[i: %i]\t{err: %g}\t{nll: %g}\t{ncomps: %i}\t{accepted: %i}\n',...
        i,err,log_f,length(unique(rho.z)),Wstats.accepted);
end

end