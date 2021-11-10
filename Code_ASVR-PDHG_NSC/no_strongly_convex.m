function output = no_strongly_convex(dataset_name, epochs, max_time)

fprintf('Test Non-Strongly-Convex methods on dataset:%s\n', dataset_name);
fprintf('epochs: %d, max_time: %ds\n', epochs, max_time);

%% Prepare dataset
fprintf('Loading dataset:%s.\n', dataset_name);
dataset_path = 'datasets/';
load([dataset_path, dataset_name, '.mat']);
if size(samples,1)~= length(labels)
    samples = samples';
end

%% Add other paths
addpath('NSC')
addpath('temp_F')
addpath('tool')

%% minibatch size for each dataset
mb=15;
ratio_train = 0.8;

%% some parameters
nu     = 1e-5;     % l1-norm
[N, d] = size(samples);
beta   = 1;
eta    = 0.01; % for stochastic methods
L = (0.25 * max(sum(samples'.^2,1)) + nu);  

%% Generate Correlation Graph
F = GetF(samples, dataset_name);

%% Stochastic training samples
idx_all       = 1:length(labels);
idx_train     = idx_all(rand(1,length(labels),1)<ratio_train);
idx_test      = setdiff(idx_all,idx_train);
train_samples = samples(idx_train,:);
test_samples  = samples(idx_test,:);
train_labels  = labels(idx_train);
test_labels   = labels(idx_test);
N_train  = length(train_labels);
clear samples labels

out_numit = N_train;
max_it = epochs * N_train;

%% Run all methods

func_list  = {@SPDHG_NSC, @SVRG_PDFP, @SVRG_ADMM_NSC, @ASVRG_ADMM_NSC, @SVR_PDHG, @ASVR_PDHG}; 
method_no = length(func_list);
fprintf('Start to run %d methods.\n', method_no);
for idx_method = 1:length(func_list)
    tic
    fprintf('Running on No.%d(Total:%d) method: %s,   ',idx_method, length(func_list), func2str(func_list{idx_method}));
    [xout{idx_method}, time{idx_method}, ind{idx_method}, iters{idx_method}] = func_list{idx_method}...
        (train_samples, train_labels, F, beta, nu, max_it, eta, mb, out_numit, max_time, L);
    
    time{idx_method} = time{idx_method};
    fprintf('time spend on: %fs.\n', toc);
end

%% Training for Class
disp('Training for Class.');
out = [];
for i = 1 : method_no
    for jj = 1: ind{i}
        out = train_samples*xout{i}(:,jj);
        flosstemp = feval(@flogistic, out, train_labels);
        floss{i}(jj) = flosstemp/size(train_samples,1)+nu*norm(F*xout{i}(:,jj),1);
    end
end
%% Test error
disp('Test error.');
out = [];
for i = 1 : method_no
    for jj = 1:ind{i}
        out = test_samples*xout{i}(:,jj);
        testtemp=feval(@flogistic, out, test_labels);
        test_err{i}(jj)=testtemp/size(test_samples,1)+nu*norm(F*xout{i}(:,jj),1); 
    end
end

%% restore loss
output.floss = floss;
output.test_err = test_err;
output.time = time;
output.xout = xout;
output.iters = iters;
%% save result
end
