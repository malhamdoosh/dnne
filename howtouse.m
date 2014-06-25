% This code shows how to run DNNE algorithm multiple times on different
% data partitions.
%
%
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


clear all;
s = RandStream('mt19937ar','Seed',1986);
for i=1:10
    % California Housing Dataset (REGRESSION)
    data = csvread('data/calhousing.data');
    indexes = randperm(s, size(data,1));
    t = ceil(0.60 * size(data,1));
    trainData = data(indexes(1:t), :);
    testData =  data(indexes(t+1:end), :);
    [housing_dnneModel{i}, housing_trnAcc(i), housing_tstAcc(i), housing_rmse(i)] = dnne(5, 50, 0.55, trainData, testData, 'reg', s);

    % German Credit Card Dataset (CLASSIFICATION)
    data = csvread('data/credit_german.data');
    indexes = randperm(s, size(data,1));
    t = ceil(0.60 * size(data,1));
    trainData = data(indexes(1:t), :);
    testData =  data(indexes(t+1:end), :);
    [credit_dnneModel{i}, credit_trnAcc(i), credit_tstAcc(i), credit_rmse(i)] = dnne(5, 100, 0.55, trainData, testData, 'class', s);
end
clear data trainData testData s indexes t i