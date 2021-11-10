% Cite
% @inproceedings{liu2017accelerated,
%   title={Accelerated variance reduced stochastic ADMM},
%   author={Liu, Yuanyuan and Shang, Fanhua and Cheng, James},
%   booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
%   volume={31},
%   number={1},
%   year={2017}
% }
function [xout, time, ind, iters] = ASVRG_ADMM_NSC(samples, labels, F, beta, nu, max_it, eta, mb, out_numit, max_time, L)
%%% Accererated SVRG-ADMM (ASVRG-ADMM) for solving general convex problems

% Initialization
[N, d] = size(samples);
x = zeros(d,1);
z = x;
y = zeros(size(F,1),1);
yold = y;
xold   = x;
zeta   = zeros(size(F,1),1); 
rnd_pm = randperm(N);
max1   = fix(max_it/(2*N));
m      = fix(2*N/mb);
theta = 0.9;
yita = 1;
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
%     yita = 1.20/(0.2 + 0.5*theta1);
    x_bar = zeros(d,1); 
    y_bar = zeros(size(F,1),1);
    y = yold;
    kk = kk + N;
%     x = (1-theta)*xold + theta*z;
    for j=1:m  
        
        % Randomly choose minibatch
        idx = ceil(N*rand);
        if idx<=N-mb+1
            ix = idx:idx+mb-1;
        else
            ix = [1:(idx+mb-N-1), idx:N];
        end
%         I = idx;
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
        y = wthresh(F*z + zeta,'s', nu/beta);        
        z = z - yita*(gg+beta*F'*(F*z - y + zeta))/theta;
        x = (1-theta)*xold + theta*z;
        zeta = zeta + F*z - y;    
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
            return;
        end
                 
    end
%     xold = x;
    xold = x_bar/m;
    yold = (1-theta)*yold + theta* y_bar/m;

    theta = (-theta^2 + sqrt(theta^4 + 4*theta^2))/2;
end    
end
