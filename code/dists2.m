function M = dists2(X1, X2, varargin)
row_insts = true;
if ~isempty(varargin)
    row_insts = varargin{1};
end
if row_insts
    X1 = X1';
    X2 = X2';
end
M = bsxfun(@plus, sum(X1 .* X1, 1)', (-2) * X1' * X2);        
M = bsxfun(@plus, sum(X2 .* X2, 1), M);        
M(M < 0) = 0;                        

end
       


