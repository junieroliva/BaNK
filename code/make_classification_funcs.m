function [funcs, names] = make_classification_funcs()

names = {'RKS', 'Bank', 'RFO', 'MKL'};
%names = {'RKS', 'Bank'};

bwb_opts.sopts.burn_in = 5;

funcs = {@(X,Y,cv,tr,ho,ts)run_bank_classification(X,Y,cv,tr,ho,ts,bwb_opts),...
         @run_RFO_classification, ...
         @run_MKL_classification};

 %funcs = {@(X,Y,cv,tr,ho,ts)run_bank_classification(X,Y,cv,tr,ho,ts,bwb_opts)};
end