% Cite
% @inproceedings{zheng2016fast,
%   title={Fast-and-Light Stochastic ADMM.},
%   author={Zheng, Shuai and Kwok, James T},
%   booktitle={IJCAI},
%   pages={2407--2613},
%   year={2016}
% }
function [xout, time, ind, iters] = SVRG_ADMM_NSC(samples, labels, F, beta, nu, max_it, eta, mb, out_numit, max_time, L)

% Initialization
[N, d] = size(samples);
x = zeros(d,1);
y = zeros(size(F,1),1);
xold   = x;
zeta   = zeros(size(F,1),1); 
rnd_pm = randperm(N);
max1   = fix(max_it/(2*N));
m      = fix(2*N/mb);
yita   = 1;
% Output
kk = 0;
ind = 1;
xout = x;
time = 0;
iters = 0;
tic
time0 = cputime;    
    
for k = 1:max1
    
    if (k > 1)
        gold = fullgrad(samples*xold, labels, 1); 
        gold = (samples'*gold)/N;
    end
    x_bar = zeros(d,1);
    y_bar = zeros(size(F,1),1);
    kk = kk + N;
    for j=1:m  
        
        % Randomly choose minibatch
        idx = ceil(N*rand);
        if idx<=N-mb+1
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1), idx:N];
        end
        I = sort(rnd_pm(ix));
        sample = samples(I,:);
        label  = labels(I);             
                
        % Gradient
        if(k==1)
            gg = feval(@minisub_grad,sample*x,label);
            gg = ((sample'*gg)/mb);
        else
            gg = feval(@minisub_grad,sample*x,label)-feval(@minisub_grad, sample*xold, label);
            gg = ((sample'*gg)/mb) + gold;        
        end        
        y = wthresh(F*x + zeta,'s', nu/beta);        
        x = x - yita*(gg+beta*F'*(F*x - y + zeta));
        zeta = zeta + F*x - y;    
        x_bar = x_bar + x; 
        y_bar = y_bar + y;
        
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
            ind = ind + 1;
            xout = [xout, x];
            time = [time, toc];
            return
        end 
                 
    end
    xold = x_bar/m;
end    
end
