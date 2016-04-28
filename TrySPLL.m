% Try SPLL
clear
clc
close all

window_size = 100;
number_of_features = 10;

[change1,pvalue1,spllstat1,change2,pvalue2,spllstat2] = deal(zeros(1,100));
for i = 1:100
    W1 = randn(window_size,number_of_features);
    W2 = randn(window_size,number_of_features);   % identical distributions
    W3 = randn(window_size,number_of_features)+2; % different distributions
    [change1(i),pvalue1(i),spllstat1(i)] = SPLL(W1,W2);
    [change2(i),pvalue2(i),spllstat2(i)] = SPLL(W1,W3);
end

%% Plots
figure,hold on
plot(change1,'k-'),plot(change2,'r-'),axis([1 100 -1 2])
title('Change'),legend('Same','Different')


figure,hold on
plot(pvalue1,'k-'),plot(pvalue2,'r-'),grid on
title('p-value'),legend('Same','Different','location','SouthEast')

figure,hold on
plot(spllstat1,'k-'),plot(spllstat2,'r-'),grid on
title('SPLL Criterion'),legend('Same','Different','location','East')
