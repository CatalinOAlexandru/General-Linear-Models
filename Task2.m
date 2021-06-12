% Permutation Testing
clc;
clear;

% Part 2.1.a
n1 = 6;
n2 = 8;
mean1 = 1.5;
mean2 = 2.0;
normDist = 0;
stdev = 0.2;

rngSeed = 20144497;
rng(rngSeed);

y1 = stdev .* randn(n1,1) + mean1 + normDist;
y2 = stdev .* randn(n2,1) + mean2 + normDist;

[h,p1,ci,stats] = ttest2(y1,y2);
tstatistic = stats.tstat;
fprintf("Part 2.1.a:\nP value is %d\nT-statistic is: %d\n",p1,tstatistic);


% Part 2.1.b.i
D = [y1;y2];


% Part 2.1.b.ii
totalPerms = 1:n1+n2;

validPerm1 = combnk(totalPerms, n1);
numPerm = length(validPerm1);
validPerm2 = zeros(numPerm,n2);


% Part 2.1.b.iii
tstatic = zeros(numPerm,1);
D1 = D(validPerm1);

for i=1:numPerm
    validPerm2(i,:) = setdiff(totalPerms, validPerm1(i,:));
    D2 = D(validPerm2(i,:));
    
    [h,p,ci,stats] = ttest2(D1(i,:),D2);
    tstatic(i) = stats.tstat;
end

% tStatHisto = histogram(tstatic,100);
% xlabel('Figure 1 - Q.2.1.b.iii Empirical distribution of all t-statistics')
% saveas(tStatHisto,'q21biii');


% Part 2.1.b.iv
pValue = nnz(tstatic >= tstatistic)/numPerm;
fprintf("\nPart 2.1.b.iv:\nP value is %d\n",pValue);


% Part 2.1.c
diffMeans = zeros(numPerm,1);
for i=1:numPerm
    validPerm2(i,:) = setdiff(totalPerms, validPerm1(i,:));
    D2 = D(validPerm2(i,:));
    
    diffMeans(i) = mean(D1(i,:)) - mean(D2); 
end

diffMeansY = mean(y1) - mean(y2);
pValueMeans = nnz(diffMeans >= diffMeansY)/numPerm;
fprintf("\nPart 2.1.c:\nP value means is %d\n",pValueMeans);

% meansHisto = histogram(diffMeans,100);
% xlabel('Figure 2 - Q.2.1.c Empirical distribution of all t-statistics')
% saveas(meansHisto,'q21c');


% Part 2.1.d.i
tStaticD = zeros(numPerm, 1);
numPerm2 = 1000;
perms = zeros(numPerm2,n1+n2);

for i=1:numPerm2
  perms(i,:) = randperm(n1+n2);
  D1 = D(perms(i,1:n1));
  D2 = D(perms(i,n1+1:end));
  [~, ~, ~, STATS]= ttest2(D1, D2);
  tStaticD(i) = STATS.tstat;
end

pValueD = nnz(tStaticD > tstatistic)/numPerm2;
fprintf("\nPart 2.1.d.i:\nP value is %d\n",pValueD);


% Part 2.1.d.iii
duplicates = 0;
for i=1:numPerm2
  for j=i+1:numPerm2
    
    a = sort(perms(i,1:n1));
    b = sort(perms(j,1:n1));
    diff1 = sum(abs(a - b));
    c = sort(perms(i,n1+1:end));
    d = sort(perms(j,n1+1:end));
    diff2 = sum(abs(c - d));

    if ((diff1 + diff2) == 0)
      duplicates = duplicates + 1;
      break;
    end
  end
end 

fprintf("\nPart 2.1.d.iii:\nNumber of duplicates: %d\n",duplicates);



