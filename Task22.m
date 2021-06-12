clc;
clear;

rngSeed = 20144497;
rng(rngSeed);

% Part 2.2

numSubjects = 8;
cpaId = [4,5,6,7,8,9,10,11];
ppaId = [3,6,9,10,13,14,15,16];
CPA = zeros(numSubjects, 40, 40, 40);
PPA = zeros(numSubjects, 40, 40, 40);

fid = fopen('glm/wm_mask.img', 'r', 'l');
data = fread(fid, 'float');
wm_mask = reshape(data, [40 40 40]);

for i=1:numSubjects
    file = sprintf('glm/CPA%d_diffeo_fa.img',cpaId(i));
    fid = fopen(file, 'r', 'l');
    data = fread(fid, 'float');
    CPA(i,:,:,:) = reshape(data, [40 40 40]);
    
    file = sprintf('glm/PPA%d_diffeo_fa.img',ppaId(i));
    fid = fopen(file, 'r', 'l');
    data = fread(fid, 'float');
    PPA(i,:,:,:) = reshape(data, [40 40 40]);
end


% Part 2.2.a
x = [repmat([1 0], numSubjects,1); repmat([0 1], numSubjects,1)];
[row,col] = size(x);
dimX = rank(x);

ContrastV = [1; -1];

tStatAll = zeros(40, 40, 40);
for i=1:40
  for j=1:40
    for k=1:40
      if (wm_mask(i,j,k) == 1)
        y = [CPA(:,i,j,k); PPA(:,i,j,k)];
        
        Px = x * pinv(x' * x) * x';
        betaHat = pinv(x' * x) * x' * y;
        Rx = eye(size(Px)) - Px;
        eHat = Rx * y;
        variance = (eHat'*eHat)/(row-dimX);
        sBeta = variance * pinv(x' * x);
        tStatAll(i,j,k) = (ContrastV'* betaHat)/(sqrt(ContrastV'*sBeta*ContrastV));
      end
    end
  end
end

save('tStatAll_q22a.mat', 'tStatAll');
maxTstat = max(tStatAll(:));
fprintf("Part 2.2.a:\nMax value of T-statistic is: %d\n",maxTstat);


% Part 2.2.b
fprintf("\nPart 2.2.b:\nRunning...\n");
n1 = 8;
n2 = 8;
sizePerm = 1:n1+n2;

validPerm1 = combnk(sizePerm, n1);
numPerm = length(validPerm1);
validPerm2 = zeros(numPerm,n2);

for i=1:numPerm
    validPerm2(i,:) = setdiff(sizePerm, validPerm1(i,:));
end

D1 = reshape(CPA, [n1 40^3])';
D2 = reshape(PPA, [n2 40^3])';
mask = reshape(wm_mask, [1 40^3]);
D = [D1, D2];


tStatAll2 = zeros(numPerm,1);
for i=1:numPerm
    newD1 = D(:,validPerm1(i,:));
    newD2 = D(:,validPerm2(i,:));
    newD = [newD1, newD2];
    
    numPerm2 = length(newD1);
    Px2 = x * pinv(x' * x) * x';
    Rx2 = eye(size(Px2)) - Px2;
    xxInv = pinv(x'*x);

    betaHat2 = pinv(x'*x)*x' * newD';
    eHat2 = Rx2 * newD';
    variance2 = sum(eHat2 .* eHat2,1)'/(row - dimX);
    xxInvRepmat = repmat(xxInv,[1,1,numPerm2]);
    invXXpermute = permute(xxInvRepmat,[3 1 2]);
    variance2repmat = repmat(variance2, [1, 2, 2]);
    sBeta2 = permute(variance2repmat .* invXXpermute, [2, 1, 3]);

    tstats2  = (ContrastV' * betaHat2) ./ sqrt(ContrastV' * [squeeze(sBeta2(1,:,:))*ContrastV, squeeze(sBeta2(2,:,:))*ContrastV]');
    tStatAll2(i) = max(tstats2 .* mask);
end

% tStatAll2Histo = histogram(tStatAll2,100);
% xlabel('Figure 3 - Q.2.2.b Empirical distribution of the maximum t-statistic')
% saveas(tStatAll2Histo,'q22b');

numPerm2 = length(D1);
Px2 = x * pinv(x' * x) * x';
Rx2 = eye(size(Px2)) - Px2;
xxInv = pinv(x'*x);

betaHat2 = pinv(x'*x)*x' * D';
eHat2 = Rx2 * D';
variance2 = sum(eHat2 .* eHat2,1)'/(row - dimX);
xxInvRepmat = repmat(xxInv,[1,1,numPerm2]);
invXXpermute = permute(xxInvRepmat,[3 1 2]);
variance2repmat = repmat(variance2, [1, 2, 2]);
sBeta2 = permute(variance2repmat .* invXXpermute, [2, 1, 3]);

tstatsP  = (ContrastV' * betaHat2) ./ sqrt(ContrastV' * [squeeze(sBeta2(1,:,:))*ContrastV, squeeze(sBeta2(2,:,:))*ContrastV]');
tStatOriginal = max(tstatsP .* mask);

fprintf("tStatOriginal is: %d\n",tStatOriginal);


% Part 2.2.c
mccPvalue = nnz(tStatAll2 > tStatOriginal)/numPerm;
fprintf("\nPart 2.2.c:\nMultiple comparisons corrected p-value is %d\n",mccPvalue);


% Part 2.2.d
tStatAll2sorted = sort(tStatAll2);
tThresh = tStatAll2sorted(floor(numPerm * 95/100));
fprintf("\nPart 2.2.d:\nMaximum t-statistic threshold is %d\n",tThresh);




