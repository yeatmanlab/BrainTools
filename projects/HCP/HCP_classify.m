read = afq.metadata.ReadEng_AgeAdj;
voc = afq.metadata.PicVocab_AgeAdj;

arcuateFA = AFQ_get(afq, 'Left Arcuate', 'fa');
arcuateMD = AFQ_get(afq, 'Left Arcuate', 'md');

lo = 90;
hi = 125;

c = hsv(7);

% Show initial scatter plot of reading/vocab
figure;
hold on;
scatter(read, voc);
xlabel('Reading Adj');
ylabel('Vocab Adj');
axis([60 160 60 160]);
line([60 160], [hi hi]);
line([60 160], [lo lo]);
line([hi hi], [60 160]);
line([lo lo], [60 160]);
lsline;

figure;
hold on;
scatter(read, nanmean(arcuateFA, 2));
xlabel('read');
ylabel('FA');
figure;
hold on;
scatter(read, nanmean(arcuateMD, 2));
xlabel('read');
ylabel('MD');
figure;
hold on;
scatter(voc, nanmean(arcuateFA, 2));
xlabel('Vocab');
ylabel('FA');
figure;
hold on;
scatter(voc, nanmean(arcuateMD, 2));
xlabel('Vocab');
ylabel('MD');

% Assign groups
% h = high, m = mid, l = low, R = reading, V = vocab

lRlV = find(read <= lo & voc <= lo);
lRmV = find(read <= lo & (voc > lo & voc <= hi));
mRlV = find((read > lo & read <= hi) & voc <= lo);
mRmV = find((read > lo & read <= hi) & (voc > lo & voc <= hi));
mRhV = find((read > lo & read <= hi) & voc > hi);
hRmV = find(read > hi & (voc > lo & voc <= hi));
hRhV = find(read > hi & voc > hi);

% Find mean FA and MD for each group
lRlVFA = nanmean(arcuateFA(lRlV, :));
lRmVFA = nanmean(arcuateFA(lRmV, :));
mRlVFA = nanmean(arcuateFA(mRlV, :));
mRmVFA = nanmean(arcuateFA(mRmV, :));
mRhVFA = nanmean(arcuateFA(mRhV, :));
hRmVFA = nanmean(arcuateFA(hRmV, :));
hRhVFA = nanmean(arcuateFA(hRhV, :));

lRlVMD = nanmean(arcuateMD(lRlV, :));
lRmVMD = nanmean(arcuateMD(lRmV, :));
mRlVMD = nanmean(arcuateMD(mRlV, :));
mRmVMD = nanmean(arcuateMD(mRmV, :));
mRhVMD = nanmean(arcuateMD(mRhV, :));
hRmVMD = nanmean(arcuateMD(hRmV, :));
hRhVMD = nanmean(arcuateMD(hRhV, :));

% Quick checkerboards
z = zeros(8, 101);
z(1:7, 1:100) = vertcat(lRlVFA, lRmVFA, mRlVFA, mRmVFA, mRhVFA, hRmVFA, hRhVFA);
FAchex = pcolor(z);


% Plot figures
figure;
hold on;
ll = scatter(lRlVFA, lRlVMD, 5, c(1,:));
lm = scatter(lRmVFA, lRmVMD, 5, c(2,:));
ml = scatter(mRlVFA, mRlVMD, 5, c(3,:));
mm = scatter(mRmVFA, mRmVMD, 5, c(4,:));
mh = scatter(mRhVFA, mRhVMD, 5, c(5,:));
hm = scatter(hRmVFA, hRmVMD, 5, c(6,:));
hh = scatter(hRhVFA, hRhVMD, 5, c(7,:));
legend([ll lm ml mm mh hm hh], 'Poor Readers, Poor Vocab', 'Poor Readers, Moderate Vocab', 'Moderate Readers, Poor Vocab', 'Moderate Readers, Moderate Vocab', 'Moderate Readers, High Vocab', 'High Readers, Moderate Vocab', 'High Readers, High Vocab');


%% FAILED SVM

% Will create a method to automatically align the data for svm training to
% see if svm can classify data accurately.

% Define reading and vocabulary age adjusted scores
read = afq.metadata.ReadEng_AgeAdj;
voc = afq.metadata.PicVocab_AgeAdj;

% Define arcuate FA and MD
arcFA = AFQ_get(afq, 'Left Arcuate', 'fa');
arcMD = AFQ_get(afq, 'Left Arcuate', 'md');

% Define thresholds for classification
lo = 90;
hi = 125;

% Assign groups
% h = high, m = mid, l = low, R = reading, V = vocab
lRlV = find(read <= lo & voc <= lo);
lRmV = find(read <= lo & (voc > lo & voc <= hi));
mRlV = find((read > lo & read <= hi) & voc <= lo);
mRmV = find((read > lo & read <= hi) & (voc > lo & voc <= hi));
mRhV = find((read > lo & read <= hi) & voc > hi);
hRmV = find(read > hi & (voc > lo & voc <= hi));
hRhV = find(read > hi & voc > hi);

% Create training matrix and group vector
arcTrain = zeros(399,200);
arcGroup = cell(1, 399);
% Set current index
ci = 1;
% Fill in matrix and vector
for ii = 1:numel(lRlV)
    arcTrain(ci, :)= horzcat(arcFA(lRlV(ii), :), arcMD(lRlV(ii), :));
    arcGroup{ci} = 'lRlV';
    ci = ci + 1;
end
for ii = 1:numel(lRmV)
    arcTrain(ci, :)= horzcat(arcFA(lRmV(ii), :), arcMD(lRmV(ii), :));
    arcGroup{ci} = 'lRmV';
    ci = ci + 1;
end
for ii = 1:numel(mRlV)
    arcTrain(ci, :)= horzcat(arcFA(mRlV(ii), :), arcMD(mRlV(ii), :));
    arcGroup{ci} = 'mRlV';
    ci = ci + 1;
end
for ii = 1:numel(mRmV)
    arcTrain(ci, :)= horzcat(arcFA(mRmV(ii), :), arcMD(mRmV(ii), :));
    arcGroup{ci} = 'mRmV';
    ci = ci + 1;
end
for ii = 1:numel(mRhV)
    arcTrain(ci, :)= horzcat(arcFA(mRhV(ii), :), arcMD(mRhV(ii), :));
    arcGroup{ci} = 'mRhV';
    ci = ci + 1;
end
for ii = 1:numel(hRmV)
    arcTrain(ci, :)= horzcat(arcFA(hRmV(ii), :), arcMD(hRmV(ii), :));
    arcGroup{ci} = 'hRmV';
    ci = ci + 1;
end
for ii = 1:numel(hRhV)
    arcTrain(ci, :)= horzcat(arcFA(hRhV(ii), :), arcMD(hRhV(ii), :));
    arcGroup{ci} = 'hRhV';
    ci = ci + 1;
end

% Train svm
arcStruct = svmtrain(arcTrain, arcGroup);

