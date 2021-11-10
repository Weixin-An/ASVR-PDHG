function [xout, time, ind, iters] = ASVR_PDHG(samples, labels, F, beta, nu, max_it, eta, mb, out_numit, max_time, L)
%% ASVRG-PDHG
% initialization
[N,d] = size(samples);
x = zeros(d,1);
z = x;
xold = x;
z_bar = xold;
y = zeros(size(F,1),1);
yold = y;

rnd_pm = randperm(N);
max1 = fix(max_it/(2*N));
m = fix(N/(4*mb));

theta = 0.9;
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

for k = 1 : max1
    x = xold;
    if(k~=1)
        zz   = samples*xold;
        gold = fullgrad(zz, labels, 1); 
        gold = samples'*gold;
        gold = gold/N;     
    end
    kk = kk + N;
    xbar = zeros(size(x));
    ybar = zeros(size(y));
    if (k == 2)
        m = fix(0.5*N/mb);
    end
    if (k > 2 && k < 10)
        m = fix(N/mb);
    end
    if (k >= 10)
        m = fix(2*N/mb);
    end
    for j=1:m
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
                   
        % Minibatch
        if(k==1)
            gg = feval(@minisub_grad, sample*x, label);
            gg = ((sample'*gg)/mb);
        else
            gg = feval(@minisub_grad, sample*x, label) - feval(@minisub_grad, sample*xold, label);
            gg = ((sample'*gg)/mb) + gold;
        end
        kk = kk + mb;
        
        % update y
        y = min(nu, max(-nu, derta*F * z_bar+y)); %y
        
        % update z&x
        temp_z = z;
        z = z - yita * (gg + F'*y)/theta;
        x = (1-theta)*xold + theta * z; % accelerate
        % acclerated averaging step
        z_bar = z + 1*(z - temp_z);      
        
        xbar = xbar + x;
        ybar = ybar + y;
        
        if toc >= max_time 
            return;
        end
    end % end inner loop
    xold = xbar/m; 
    yold = (1-theta)*yold + theta * (ybar/m);
    theta = (-theta^2 + sqrt(theta^4 + 4*theta^2))/2;
    
    % append out
    ind = ind + 1;
    xout(:,ind) = x;
    time(ind) = toc;
    iters = [iters, kk/N];

end
xout = xout(:,1:ind);
time = time(1:ind);

end
