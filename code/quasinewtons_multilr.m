function [beta, obj] = quasinewtons_multilr(PhiW,Y,c,beta_curr,varargin)
if ~isempty(varargin)
    opts = varargin{1};
else
    opts = struct;
end

optiopts = optimoptions(@fminunc,'GradObj','on','TolFun',1E-10,...
          'MaxIter',10000, 'Display', 'off', 'Algorithm', 'quasi-newton');
optiopts = get_opt(opts,'optiopts',optiopts);

funcs = newtons_multilr_funcs(PhiW,Y,c,beta_curr,opts);
[beta,obj] = fminunc(@(x)multi_output(x,funcs{1},funcs{2}),beta_curr,optiopts);

    function [objv,gradv] = optifunc(xv)
        objv = funcs{1}(xv);
        gradv = funcs{2}(xv);
    end

end