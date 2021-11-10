function F = GetF(samples, dataset_name)
[N, d] = size(samples);
if ( exist(['F_' dataset_name '.mat'],'file') ==0 )
    
    S = cov(samples);
    [W, invW, adj] = graphical_lasso(S, 0.005, 1/3, 10, eye(d));
    adj = zeros(size(invW));
    adj(abs(invW) > 2.5e-3) = 1;    
    F = -tril(adj,-1) + triu(adj,1);
    F = [F;eye(d)];
    save(['temp_F/F_' dataset_name '.mat'], 'F');
else
    load(['F_' dataset_name '.mat'], 'F');
end
clear W invW adj
