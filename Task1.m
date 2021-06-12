% General Linear Model
clear;
clc;

% Part 1.1.a
sampleSize = 20;
mean1 = 1.5;
mean2 = 2.0;
normDist = 0;
stdev = 0.2;

rngSeed = 20144497;
rng(rngSeed);

y1 = stdev .* randn(sampleSize,1) + mean1 + normDist;
y2 = stdev .* randn(sampleSize,1) + mean2 + normDist;
estimatedMu1 = mean(y1);
estimatedMu2 = mean(y2);
estimatedStd1 = std(y1);
estimatedStd2 = std(y2);

tolerance = 0.01;
checkEstimates = [(estimatedMu1 - mean1)^2 < tolerance,...
                  (estimatedMu2 - mean2)^2 < tolerance,...
                  (estimatedStd1 - stdev)^2 < tolerance,...
                  (estimatedStd2 - stdev)^2 < tolerance];
if ~(all(checkEstimates))
    error("Estiamtes are wrong.");
else
    fprintf('Part 1.1.a:\nResults verified as correct with tolerance of %f\n',tolerance);
end


% Part 1.1.b
[h,p,ci,stats] = ttest2(y1,y2);
if (h == 0)
    error('Failed to reject the null hypothesis.');
else
    fprintf("\nPart 1.1.b:\nNull hypothesis rejected.\np is %d\n",p);
    disp(stats);
end


% Part 1.1.c.i
x = [repmat([0 1], sampleSize,1); repmat([1 0], sampleSize,1)];
[row,col] = size(x);
dimX = rank(x);
fprintf("Part 1.1.c.i:\nRows: %d\nColumns: %d\nRank: %d\n",row,col,dimX);


% Part 1.1.c.ii
Px = x * inv(x' * x) * x';
tracePx = trace(Px);
fprintf("\nPart 1.1.c.ii:\nTrace of Px: %d\n",tracePx);

% Part 1.1.c.iii
y = [y1;y2];
yHat = Px * y;


% Part 1.1.c.iv
Rx = eye(size(Px)) - Px;


% Part 1.1.c.v
eHat = Rx * y;
dimCXperp = trace(Rx);
fprintf("\nPart 1.1.c.v:\nDimension of C(x) perpedicular is %d\n",dimCXperp);


% Part 1.1.c.vi
theta = acosd(eHat' * yHat / (norm(eHat) * norm(yHat)));
fprintf("\nPart 1.1.c.vi:\nAngle is %f\n",theta);


% Part 1.1.c.vii
betaHat = inv(x' * x) * x' * y;
fprintf("\nPart 1.1.c.vii:\nBeta hat is:\n");
disp(betaHat);


% Part 1.1.c.viii
variance = (eHat'*eHat)/(row-dimX);
fprintf("Part 1.1.c.viii:\nVariance is: %d\n",variance);


% Part 1.1.c.ix
sBeta = variance * inv(x' * x);
fprintf("\nPart 1.1.c.ix:\nCovariance S_beta is:\n");
disp(sBeta);

sBetaSTD = sqrt(sBeta);
fprintf("STD of model parameter is:\n");
disp(sBetaSTD);


% Part 1.1.c.x 
ContrastV = [1 -1]';
NullV = null(ContrastV');
ReducedMod = x * NullV;


% Part 1.1.c.xi
Px0 = (ReducedMod * inv(ReducedMod' * ReducedMod)) * ReducedMod' ;
Rx0 = eye(size(Px0)) - Px0;
eHatX0 = Rx0 * y;

% Calculate F Statistic
varianceX = eHat'*eHat/(length(y)-rank(Px));
dimDiff = (length(y)-rank(Px0)) - (length(y)-rank(Px));
eIncrease = ((eHatX0'*eHatX0)-(eHat'*eHat))/dimDiff;
fStatic = eIncrease/varianceX;

fprintf("Part 1.1.c.xi:\nF-Static is: %d\n",fStatic);


% Part 1.1.c.xii
tStatic = (ContrastV'* betaHat)/(sqrt(ContrastV'*sBeta*ContrastV));
fprintf("\nPart 1.1.c.xii:\nT-Static is: %d\n",tStatic);


% Part 1.1.c.xiv
truth = [2.0; 1.5];
eTrue = y - x*truth;
e = Px * (y - x*truth);


% Part 1.1.c.xv
projectionEerror = Rx * eTrue;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Part 1.1.d.i
x2 = [repmat([1 0 1], sampleSize,1); repmat([1 1 0], sampleSize,1)];
[rowX2,colX2] = size(x2);
dimX2 = rank(x2);
fprintf("\nPart 1.1.d.i:\nRows: %d\nColumns: %d\nRank: %d\n",rowX2,colX2,dimX2);


% Part 1.1.d.ii
Px2 = x2 * pinv(x2' * x2) * x2';
tracePx = trace(Px2);
fprintf("\nPart 1.1.d.ii:\nTrace of Px: %d\n",tracePx);


% Part 1.1.d.iii
ContrastV2 = [0 1 -1]';
NullV2 = null(ContrastV2');
ReducedMod2 = x2 * NullV2;


% Part 1.1.d.iv
betaHat2 = pinv(x2' * x2) * x2' * y;
Rx2 = eye(size(Px2)) - Px2;
eHat2 = Rx2 * y;
variance2 = (eHat2'*eHat2)/(rowX2-dimX2);
sBeta2 = variance2 * pinv(x2' * x2);
tStatic2 = (ContrastV2'* betaHat2)/(sqrt(ContrastV2'*sBeta2*ContrastV2));

fprintf("\nPart 1.1.d.iv:\nT-Static is: %d\n",tStatic2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Part 1.1.e.i
x3 = [repmat([1 0], sampleSize,1); repmat([1 1], sampleSize,1)];
[rowX3,colX3] = size(x3);
dimX3 = rank(x3);
fprintf("\nPart 1.1.e.i:\nRows: %d\nColumns: %d\nRank: %d\n",rowX3,colX3,dimX3);


% Part 1.1.e.ii
ContrastV3 = [0 1]';
NullV3 = null(ContrastV3');
ReducedMod3 = x3 * NullV3;


% Part 1.1.e.iii
Px3 = x3 * pinv(x3' * x3) * x3';
betaHat3 = pinv(x3' * x3) * x3' * y;
Rx3 = eye(size(Px3)) - Px3;
eHat3 = Rx3 * y;
variance3 = (eHat3'*eHat3)/(rowX3-dimX3);
sBeta3 = variance3 * pinv(x3' * x3);
tStatic3 = (ContrastV3'* betaHat3)/(sqrt(ContrastV3'*sBeta3*ContrastV3));

fprintf("\nPart 1.1.e.iii:\nT-Static is: %d\n",tStatic3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Part 1.2.a
[h1,p1,ci1,stats1] = ttest(y1,y2);
[h2,p2,ci2,stats2] = ttest2(y1,y2);
fprintf("\nPart 1.2.a: \nttest:\np: %d\ntstat: %d\n",p1,stats1.tstat);
fprintf("\nttest2:\np: %d\ntstat: %d\n",p2,stats2.tstat);


% Part 1.2.b.i
x4 = [repmat([1 0], sampleSize,1); repmat([1 1], sampleSize,1)];
x4 = [x4, [eye(sampleSize);eye(sampleSize)]];
[rowX4,colX4] = size(x4);
dimX4 = rank(x4);
fprintf("\nPart 1.2.b.i:\nRank is: %d\n",dimX4);


% Part 1.2.b.ii
ContrastV4 = zeros(sampleSize + 2, 1);
ContrastV4(2) = 1;
NullV4 = null(ContrastV4');
ReducedMod4 = x4 * NullV4;


% Part 1.2.b.iii
Px4 = x4 * pinv(x4' * x4) * x4';
betaHat4 = pinv(x4' * x4) * x4' * y;
Rx4 = eye(size(Px4)) - Px4;
eHat4 = Rx4 * y;
variance4 = (eHat4'*eHat4)/(rowX4-dimX4);
sBeta4 = variance4 * pinv(x4' * x4);
tStatic4 = (ContrastV4'* betaHat4)/(sqrt(ContrastV4'*sBeta4*ContrastV4));

fprintf("\nPart 1.2.b.iii:\nT-Static is: %d\n",tStatic4);



