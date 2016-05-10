load datasets/bike.mat

f = make_regression_funcs();
opts.funcs = f;
method_stats = run_regression(X,(Y-mean(Y))./std(Y),opts);