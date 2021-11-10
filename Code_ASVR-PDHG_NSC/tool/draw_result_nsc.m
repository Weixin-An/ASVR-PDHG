function draw_result_nsc(output)
floss = output.floss;
time = output.time;
iter = output.iters;
%% Plot Results
aaa1=min(floss{1}(:)); aaa2=min(floss{2}(:)); aaa3=min(floss{3}(:)); aaa4=min(floss{4}(:)); aaa5=min(floss{5}(:));aaa6=min(floss{6}(:));
minval = min([aaa1,aaa2,aaa3, aaa4, aaa5, aaa6]) - 5e-8;

%% training loss Vs. time
h=figure;semilogy(time{1}(1:2:end), abs(floss{1}(1:2:end) - minval(1:2:end)),'g--^'); 
hold on, semilogy(time{2}(1:2:end), abs(floss{2}(1:2:end) - minval(1:2:end)),'y--s'); 
hold on, semilogy(time{3}(1:2:end), abs(floss{3}(1:2:end) - minval(1:2:end)),'c--^'); 
hold on, semilogy(time{4}(1:2:end), abs(floss{4}(1:2:end) - minval(1:2:end)),'m--^');  
hold on, semilogy(time{5}(1:2:end), abs(floss{5}(1:2:end) - minval(1:2:end)),'b--^');
hold on, semilogy(time{6}(1:2:end), abs(floss{6}(1:2:end) - minval(1:2:end)),'r--^');
hold off

xlabel('CPU times (s)');
ylabel('Objective minus best');
legend('SPDHG', 'SVRG-PDFP', 'SVRG-ADMM', 'ASVRG-ADMM','SVR-PDHG', 'ASVR-PDHG');
axis([0, 1600, 1e-4, 1])



