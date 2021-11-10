% Demo for Non-Strongly Convex Case
clear all;
dataset_name = 'epsilon_test';

epochs = 30; 
max_time = 5000;

output = no_strongly_convex(dataset_name, epochs, max_time); 
%% Plot Results
draw_result_nsc(output);