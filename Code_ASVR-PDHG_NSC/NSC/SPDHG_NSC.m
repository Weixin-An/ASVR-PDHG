function [xout, time, ind, iters] = SPDHG_NSC(samples, labels, F, beta, nu, max_it, eta, mb, out_numit, max_time, L)
%% S_PDHG
% initialization
s      = 5e-5; 
iters = 0;
[N,d]  = size(samples);
x      = zeros(d,1);
xbar   = x;
y      = zeros(size(F,1),1);
rnd_pm = randperm(N);
max1 = max_it/mb;

% out
kk = 0;
ind = 1;
xout = x;
time = 0;
tic
time0 = cputime;    

for k=1:max1
    yita = 1 / (sqrt(k)+L);
    % randomly choose samples
    idx = ceil(N*rand);
    if idx <= N-mb+1
        ix = idx:idx+mb-1;
    else
        ix = [1:(idx+mb-N-1), idx:N];
    end
    I = sort(rnd_pm(ix));
    sample = samples(I,:);
    label  = labels(I); 
    
    g = feval(@minisub_grad,sample*x,label);
    g = (sample'*g)/mb; 
    
    % update y
    y = min(nu, max(-nu, s*F * x+y));
    % update x
    x = x - yita*(g + F'*y);
    xbar = ((k - 1) * xbar + x)/k;
    
    % Output
    kk = kk+mb;
    if kk >= out_numit
        iters = [iters, ind];
        ind = ind + 1;
        xout = [xout, x];
        time = [time, toc];
        kk = kk -out_numit;
    end
    
    if toc >= max_time
        return;
    end
end

end