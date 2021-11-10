function [xout, time, ind, iters] = SVRG_PDFP(samples, labels, F, beta, nu, max_it, eta, mb, out_numit, max_time, L)
%% VR-SPDHG
% initialization
[N,d] = size(samples);
x = zeros(d,1);
x0 = x; 
xold = x; 
v = zeros(size(F,1),1);
lambda = 1e-5;
gamma = 1;

rnd_pm = randperm(N);
max1 = fix(max_it/(2*N));
m = fix(N/mb);

yita = 1; 
derta = 1; 

% out
kk = 0;
ind = 1;
xout = zeros(size(x,1),100);
time = zeros(1,100);
iters = 0;
tic
time0 = cputime;   

for k = 1:max1
    if(k~=1)
        zz   = samples*xold;
        gold = fullgrad(zz, labels, 1); 
        gold = samples'*gold;
        gold = gold/N; 
    end
    kk = kk + N;
    xbar = zeros(size(x));
    vbar = zeros(size(v));
    for j=1:m
        % randomly choose samples:        
        idx = ceil(N*rand);
        if idx <= N-mb+1 
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1), idx:N];
        end
        I = sort(rnd_pm(ix)); 
        sample = samples(I,:);
        label  = labels(I);              
        % Minibatch
        if(k==1)
            gg = feval(@minisub_grad, sample*x, label); 
            gg = ((sample'*gg)/mb); 
        else 
            gg = feval(@minisub_grad, sample*x, label) - feval(@minisub_grad, sample*xold, label);
            gg = ((sample'*gg)/mb) + gold;
        end
        kk = kk + mb;
        % update x
        y = x - gamma * gg - gamma * F' * v;
        % update v
%         v = wthresh(lambda/gamma * F * y + v, 's', lambda/gamma);
        v = min(nu, max(-nu, gamma * F * y + v));
        x = x - gamma * (gg + F'*v);
        
        xbar= xbar + x;
        % vbar= vbar + v;
        
        if toc >= max_time
            xout = xout(:, 1:ind);
            time = time(1:ind);
            return;
        end     
    end
    xold = xbar/m;  
    ind = ind + 1;
    xout(:, ind) = xold; 
    time(ind) = toc;
    iters = [iters, kk/N];
end
xout = xout(:, 1:ind);
time = time(1:ind);
end

