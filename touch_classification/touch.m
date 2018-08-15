%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

X = load('touch_log.txt');
[m, n] = size(X);
var = std(X, 0, 1) .^ 2;
var = repmat(var, m, 1);

T = load('test.txt');
T = repmat(T, m, 1);

Y = (X - T) ./ var;

Y = sum(Y, 2);