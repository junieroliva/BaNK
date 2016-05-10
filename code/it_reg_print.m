function it_reg_print(i,data,rho,Wstats,varargin)

if ~isempty(varargin)
    opts = varargin{end};
else
    opts = struct;
end
mod_it = get_opt(opts,'mod_it',1);

if mod(i,mod_it)==0
    Y = data{2};
    if isfield(Wstats,'PhiWbeta')
        MSE = mean(mean((Y-Wstats.PhiWbeta).^2));
    else
        MSE = mean(mean((Y-Wstats.PhiW*Wstats.betaW).^2));
    end

    if ~isfield(Wstats,'accepted')
        fprintf('[i: %i]\t{MSE: %g}\t{ncomps: %i}\n',...
            i,MSE,length(unique(rho.z)));
    else
        fprintf('[i: %i]\t{MSE: %g}\t{ncomps: %i}\t{accepted: %i}\n',...
            i,MSE,length(unique(rho.z)),Wstats.accepted);
    end
end

end