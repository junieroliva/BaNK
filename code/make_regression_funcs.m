function [funcs, names] = make_regression_funcs()

names = {'RKS', 'BaNK_marg'};

funcs = {@run_bank_regression};

end