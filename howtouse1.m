% This software package has been developed by Monther Alhamdoosh (c) 2014
% based on this paper
% Monther Alhamdoosh, Dianhui Wang, Fast decorrelated neural network ensembles
% with random weights, Information Sciences, Volume 264, 20 April 2014, 
% Pages 104-117, ISSN 0020-0255, http://dx.doi.org/10.1016/j.ins.2013.12.016.
%
% For technical support and/or help, please contact m.hamdoosh@gmail.com
%
% This package has been downloaed from http://homepage.cs.latrobe.edu.au/dwang/
%
% Computer Activity Dataset (REGRESSION)
clear all;
data = csvread('data/computer_activity.data');
X = data(:, 2:end);
T = data(:, 1);

s = RandStream('mt19937ar','Seed',54829);    
RandStream.setGlobalStream(s);

dnne = newdnne(5, 70, X, T, 0.5);

[dnne, rmse] = traindnne(dnne, X, T);

netOut = simdnne(dnne, X);

rmse1 = sqrt(sum((T - netOut).^2) / size(T,1));