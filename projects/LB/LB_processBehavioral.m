cd /biac4/wandell/data/Lindamood_Bell/data/
[~,~,data] = xlsread('Behavioral_Data_Spreadsheet.xls');
towreS = [59 63; 87 91; 116 120; 144 148];
word4 = [69 97 126 154];
wj = [74 102 131 159];
subCols = [2 3 4 11 15 16];
c1=0;c2=0;
varNames = {'TOWRE sight word raw', 'WJ Word ID raw', '4 Letter word list'};
yNames = {'# words read in 45s' '# words read accurately' '# words read in 45s'}
for ii = subCols
   c1=c1+1;c2=0;
   for jj = 1:4
      c2=c2+1;
      score(c1,c2,1) = nanmean(vertcat(data{towreS(jj,:),ii}));
      score(c1,c2,2) = data{word4(jj),ii};
      score(c1,c2,3) = data{wj(jj),ii};
   end 
end
%% Plot each subject
figure;
subplot(1,3,1);plot(score(:,:,1)','linewidth',3);axis tight
title(varNames{1}); ylabel(yNames{1}); xlabel('Session')
subplot(1,3,2);plot(score(:,:,2)','linewidth',3);axis tight
title(varNames{2}); ylabel(yNames{2}); xlabel('Session')
subplot(1,3,3);plot(score(:,:,3)','linewidth',3);axis tight
title(varNames{3}); ylabel(yNames{3}); xlabel('Session')

%% Make plots
nplot = size(score,3);
usecols = 1:3;
figure;
for ii = 1:size(score,3)
   % demean each row and then add back the grand mean
   m1 = nanmean(flatten(score(:,usecols,ii)));
   m2 = nanmean(score(:,usecols,ii),2);
   d1 = score(:,usecols,ii) - repmat(m2,[1 length(usecols)]) + repmat(m1, [size(score,1) length(usecols)]);
   % Calculate the mean and standard error for each measurement point
   session_means = nanmean(d1,1);
   session_se    = nanstd(d1)./sqrt(size(d1,1));
   % Plot
   subplot(1,nplot,ii);
   errorbar(session_means,session_se,'-ko','markerfacecolor','k')
   title(varNames{ii}); ylabel(yNames{ii}); xlabel('Session')
   set(gca,'xtick',usecols)
end