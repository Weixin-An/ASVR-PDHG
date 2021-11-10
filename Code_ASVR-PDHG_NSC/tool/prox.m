function [ result ] = prox( y, threshold )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
result = y;
index1 = find(y >= threshold);
index2 = find(y < threshold & y > -threshold);
index3 = find(y <= -threshold);
result(index1) = y(index1) - threshold;
result(index2) = 0;
result(index3) = y(index3) + threshold;
end

