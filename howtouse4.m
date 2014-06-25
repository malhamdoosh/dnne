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
% LED 7 Dataset (MULTI-CLASS CLASSIFICATION)
clear all;
data = csvread('data/led_7.data');
X = data(:, 2:end);
TOrig = data(:, 1);
if min(TOrig) == 1
    noClasses = max(TOrig);
    T = ones(size(TOrig,1), noClasses) * -1;
    for i=1:noClasses
        T(TOrig == i, i) = 1;
    end
elseif min(TOrig) == 0
    noClasses = max(TOrig) + 1;
    T = ones(size(TOrig,1), noClasses) * -1;
    for i=1:noClasses
        T(TOrig == i - 1, i) = 1;
    end
end

clear noClasses data i;

s = RandStream('mt19937ar','Seed',54829);    
RandStream.setGlobalStream(s);
   
dnne = newdnne(5, 50, X, T, 0.55);

[dnne, rmse] = traindnne(dnne, X, T);

predLabels = simdnne(dnne, X, 'class');
if min(TOrig) == 1
    predLabels = simdnne(dnne, X, 'class');
elseif min(TOrig) == 0
    predLabels = simdnne(dnne, X, 'class') - 1;
end

acc = sum(TOrig == predLabels) / size(TOrig,1) * 100;