function it_mclass_print(i,data,rho,Wstats,~,model_prior)
Y = data{2};
[N,C] = size(Y);

beta = Wstats.mode;
if isfield(Wstats,'PhiWbeta')
    PhiWbeta = Wstats.PhiWbeta;
else
    PhiWbeta = Wstats.PhiW*reshape(beta,[],C);
end

[~,pred_classes] = max(PhiWbeta,[],2);
Y_pred = zeros(N,C);
Y_pred(sub2ind([N,C],(1:N)',pred_classes)) = 1;

err = 1-mean(Y_pred(Y>0));
M = min(PhiWbeta, [], 2);
PhiWBt_s = bsxfun(@minus, PhiWbeta, M);
log_f = -sum(sum(Y.*PhiWbeta)) + sum(log( sum(exp(PhiWBt_s),2) )) + sum(M) + 0.5*model_prior.c*sum(beta.^2);

if ~isfield(Wstats,'accepted')
    fprintf('[i: %i]\t{err: %g}\t{nll: %g}\t{ncomps: %i}\n',...
        i,err,log_f,length(unique(rho.z)));
else
    fprintf('[i: %i]\t{err: %g}\t{nll: %g}\t{ncomps: %i}\t{accepted: %i}\n',...
        i,err,log_f,length(unique(rho.z)),Wstats.accepted);
end

end