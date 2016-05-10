addpath(genpath('/usr0/home/akdubey/Juice/mtimesx'));
addpath(genpath('/usr0/home/akdubey/Juice/minFunc_2012'));
addpath(genpath('/usr0/home/akdubey/Juice/funcLearn'));
addpath(genpath('/usr0/home/akdubey/Juice/gpml-matlab-v3.6-2015-07-07'));
addpath(genpath('/usr0/home/akdubey/Juice/lbfgs/lbfgsb-matlab/src'));
addpath(genpath('/usr0/home/akdubey/Juice/projectbirdup/kernelLearn/code'));

[funcs, names] = make_classification_funcs();

cv_opts.names = names;
cv_opts.K = 5;
cv_opts.fileprefix = 'results_checking';
cv_opts.stand_response = false;
load ../../datasets/Classification/pima_data.mat
parpool(3);
[tst_errs, tst_resids, method_stats, inds, hol_inds] = kfold_cv(X,Y,@run_classification,funcs,cv_opts);
