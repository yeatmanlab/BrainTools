function HCP_arcuateData(afq)

% Setting up a series of analyses and plots to focus only on multiple
% variables of the arcuate with regards to the other data

readRaw = afq.metadata.ReadEng_Unadj;
vocabRaw = afq.metadata.PicVocab_Unadj;
readAdj = afq.metadata.ReadEng_AgeAdj;
vocabAdj = afq.metadata.PicVocab_AgeAdj;

arcuateFA = AFQ_get(afq, 'Left Arcuate', 'fa');
arcuateMD = AFQ_get(afq, 'Left Arcuate', 'md');

figure;
hold on;
subplot(1,2,1);
scatter(readRaw, vocabRaw);
xlabel('Reading Unadj');
ylabel('Vocab Unadj');
subplot(1,2,2);
scatter(readAdj, vocabAdj);
xlabel('Reading Adj');
ylabel('Vocab Adj');

g = readAdj>85;
m1 = nanmean(arcuateFA(g==1, :));
m2 = nanmean(arcuateFA(g~=1, :));
m3 = nanmean(arcuateMD(g==1, :));
m4 = nanmean(arcuateMD(g~=1, :));


se1 = nanstd(arcuateFA(g==1, :))./sqrt(size(arcuateFA(g==1, :),1));
se2 = nanstd(arcuateFA(g~=1, :))./sqrt(size(arcuateFA(g~=1, :),1));
se3 = nanstd(arcuateMD(g==1, :))./sqrt(size(arcuateMD(g==1, :),1));
se4 = nanstd(arcuateMD(g~=1, :))./sqrt(size(arcuateMD(g~=1, :),1));

figure;
hold on;
scatter(m1,m3,5,[1 0 0]);
scatter(m2,m4,5,[0 0 1]);