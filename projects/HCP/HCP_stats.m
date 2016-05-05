% function HCP_stats(afq, varargin) %Uncomment to turn into a function,
% just using to set up stats for now...

% Will be built out to automatically calculate stats between different
% aspects of data processed by AFQ and behavioral measures collected by
% HCP.
%
%

% Set up all variables (for arcuate)
bxRead = vertcat(afq.metadata.ReadEng_Unadj);
bxVocab = vertcat(afq.metadata.PicVocab_Unadj);
AF_rd = afq.vals.rd{19}';
AF_fa = afq.vals.fa{19}';
AF_md = afq.vals.md{19}';
AF_ad = afq.vals.ad{19}';
AF_cl = afq.vals.cl{19}';

% Generate plot for each factor
figure;
hold on;
F1 = scatter(bxRead, bxVocab);
xlabel('Reading');
ylabel('Vocab');

figure;
hold on;
F2 = pcolor(AF_rd);
title('rd of the arcuate');
colorbar;

figure;
hold on;
F3 = pcolor(AF_fa);
title('fa of the arcuate');
colorbar;

figure;
hold on;
F4 = pcolor(AF_md);
title('md of the arcuate');
colorbar;

figure;
hold on;
F5 = pcolor(AF_ad);
title('ad of the arcuate');
colorbar;

figure;
hold on;
F6 = pcolor(AF_cl);
title('cl of the arcuate');
colorbar;

% Calculate means
mnAF_rd = mean(AF_rd)';
mnAF_fa = mean(AF_fa)';
mnAF_md = mean(AF_md)';
mnAF_ad = mean(AF_ad)';
mnAF_cl = mean(AF_cl)';

% Extract medians
mdAF_rd = AF_rd(50,:)';
mdAF_fa = AF_fa(50,:)';
mdAF_md = AF_md(50,:)';
mdAF_ad = AF_ad(50,:)';
mdAF_cl = AF_cl(50,:)';

% Run correlations
vars = horzcat(bxRead, bxVocab, mnAF_ad, mnAF_cl, mnAF_fa, mnAF_md, mnAF_rd);
[~, numvars] = size(vars);
corrs = cell(numvars);

mdvars = horzcat(bxRead, bxVocab, mdAF_ad, mdAF_cl, mdAF_fa, mdAF_md, mdAF_rd);
mdcorrs = cell(numvars);

for ii = 1:numvars
    for jj = 1:numvars
        corrs{ii,jj} = corr(vars(:,ii), vars(:,jj), 'rows', 'pairwise');
    end
end

for ii = 1:numvars
    for jj = 1:numvars
        mdcorrs{ii,jj} = corr(mdvars(:,ii), mdvars(:,jj), 'rows', 'pairwise');
    end
end

corrT = cell2table(corrs, 'VariableNames', {'bxRead' 'bxVocab' 'mnAF_ad' 'mnAF_cl' 'mnAF_fa' 'mnAF_md' 'mnAF_rd'}, 'RowNames', {'bxRead' 'bxVocab' 'mnAF_ad' 'mnAF_cl' 'mnAF_fa' 'mnAF_md' 'mnAF_rd'});

mdcorrT = cell2table(mdcorrs, 'VariableNames', {'bxRead' 'bxVocab' 'mdAF_ad' 'mdAF_cl' 'mdAF_fa' 'mdAF_md' 'mdAF_rd'}, 'RowNames', {'bxRead' 'bxVocab' 'mdAF_ad' 'mdAF_cl' 'mdAF_fa' 'mdAF_md' 'mdAF_rd'});

%output into Table
T = table(mnAF_ad, mnAF_cl, mnAF_fa, mnAF_md, mnAF_rd, bxRead);
md1 = fitlm(T);
md1step = stepwiselm(T);

T2 = table(mdAF_ad, mdAF_cl, mdAF_fa, mdAF_md, mdAF_rd, bxRead);
md2 = fitlm(T2);
md2step = stepwiselm(T2);

% T.Properties.VariableNames = names;
% T.RowNames = names;

corrT
md1
md1step
mdcorrT
md2
md2step

%% Making some figures

% Scatter plots
for fgnum = 1:20;
    fgnames = AFQ_get(afq, 'fgnames');
    r = AFQ_get(afq, 'metadata', 'ReadEng_AgeAdj');
    fa = AFQ_get(afq, fgnames{fgnum}, 'fa');
    md = AFQ_get(afq, fgnames{fgnum}, 'md');
    % plot them out
    figure;
    subplot(1,2,1);
    plot(mean(fa, 2), r, 'ko', 'markerfacecolor', [.8 0 0]);lsline
    xlabel(sprintf('%s (tract num %d) FA', fgnames{fgnum}, fgnum),...
        'fontsize', 14, 'fontname', 'times')
    ylabel('Reading score','fontsize', 14, 'fontname', 'times')
    subplot(1,2,2);
    plot(mean(md, 2), r, 'ko', 'markerfacecolor', [0 0 .8]);lsline
    xlabel(sprintf('%s (tract num %d) MD', fgnames{fgnum}, fgnum),...
        'fontsize', 14, 'fontname', 'times')
    ylabel('Reading score','fontsize', 14, 'fontname', 'times')
    
    % Average tract profiles
    g   = r>85;
    m1  = nanmean(fa(g==1, :));
    m2  = nanmean(fa(g==0, :));
    % NOTE THAT WE SHOULD CALCULATE SE BASED ON THE NUMBER OF NON NAN SUBS
    se1 = nanstd(fa(g==1, :))./sqrt(size(fa(g==1, :),1));
    se2 = nanstd(fa(g==0, :))./sqrt(size(fa(g==0, :),1));
    % plot them
    figure; hold('on');
    patch([1:100 fliplr(1:100)],[m1 - se1, fliplr(m1 + se1)],[.8 0 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
    h(1) = plot(1:100, m1, '-', 'color', [.8 0 0], 'linewidth',3);
    title(fgnames(fgnum));
    patch([1:100 fliplr(1:100)],[m2 - se2, fliplr(m2 + se2)],[0 0 .8],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
    h(2) =  plot(1:100, m2, '-', 'color', [0 0 .8], 'linewidth',3);
    title(fgnames(fgnum));
    legend(h,'Good readers', 'Poor readers');
    
end

%% Age Adjusted,

% Scatter plots
for fgnum = 1:20;
    fgnames = AFQ_get(afq, 'fgnames');
    r = AFQ_get(afq, 'metadata', 'ReadEng_AgeAdj');
    fa = AFQ_get(afq, fgnames{fgnum}, 'fa');
    md = AFQ_get(afq, fgnames{fgnum}, 'md');
    % plot them out
    figure;
    subplot(1,2,1);
    plot(mean(fa, 2), r, 'ko', 'markerfacecolor', [.8 0 0]);lsline
    xlabel(sprintf('%s (tract num %d) FA', fgnames{fgnum}, fgnum),...
        'fontsize', 14, 'fontname', 'times')
    ylabel('Reading score','fontsize', 14, 'fontname', 'times')
    subplot(1,2,2);
    plot(mean(md, 2), r, 'ko', 'markerfacecolor', [0 0 .8]);lsline
    xlabel(sprintf('%s (tract num %d) MD', fgnames{fgnum}, fgnum),...
        'fontsize', 14, 'fontname', 'times')
    ylabel('Reading score','fontsize', 14, 'fontname', 'times')
    
    % Average tract profiles
    h   = r>=120;
    p   = r<=80;
    g   = r>80 & r<120;
    m1  = nanmean(fa(g==1, :));
    m2  = nanmean(fa(p==1, :));
    m3  = nanmean(fa(h==1, :));
    

    % NOTE THAT WE SHOULD CALCULATE SE BASED ON THE NUMBER OF NON NAN SUBS
    se1 = nanstd(fa(g==1, :))./sqrt(size(fa(g==1, :),1));
    se2 = nanstd(fa(p==1, :))./sqrt(size(fa(p==1, :),1));
    se3 = nanstd(fa(h==1, :))./sqrt(size(fa(h==1, :),1));
    

    % plot them
    figure; hold('on');
    patch([1:100 fliplr(1:100)],[m1 - se1, fliplr(m1 + se1)],[.8 0 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
    plot(1:100, m1, '-', 'color', [.8 0 0], 'linewidth',3);
    patch([1:100 fliplr(1:100)],[m2 - se2, fliplr(m2 + se2)],[0 0 .8],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
    plot(1:100, m2, '-', 'color', [0 0 .8], 'linewidth',3);
    patch([1:100 fliplr(1:100)],[m3 - se3, fliplr(m3 + se3)],[0 .8 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
    plot(1:100, m3, '-', 'color', [0 .8 0], 'linewidth',3);

    md1  = nanmean(md(g==1, :));
    md2  = nanmean(md(p==1, :));
    md3  = nanmean(md(h==1, :));

    semd1 = nanstd(md(g==1, :))./sqrt(size(md(g==1, :),1));
    semd2 = nanstd(md(p==1, :))./sqrt(size(md(p==1, :),1));
    semd3 = nanstd(md(h==1, :))./sqrt(size(md(h==1, :),1));

    figure; hold('on');
    patch([1:100 fliplr(1:100)],[md1 - semd1, fliplr(md1 + semd1)],[.8 0 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
    plot(1:100, md1, '-', 'color', [.8 0 0], 'linewidth',3);
    patch([1:100 fliplr(1:100)],[md2 - semd2, fliplr(md2 + semd2)],[0 0 .8],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
    plot(1:100, md2, '-', 'color', [0 0 .8], 'linewidth',3);
    patch([1:100 fliplr(1:100)],[md3 - semd3, fliplr(md3 + semd3)],[0 .8 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
    plot(1:100, md3, '-', 'color', [0 .8 0], 'linewidth',3);
end


%% Isolated mean of Arcuate tracts at place of greatest diversion


    fgnames = AFQ_get(afq, 'fgnames');
    r = AFQ_get(afq, 'metadata', 'ReadEng_AgeAdj');
    fa = AFQ_get(afq, fgnames{fgnum}, 'fa');
    md = AFQ_get(afq, fgnames{fgnum}, 'md');

    % Average tract profiles
    h   = r>=120;
    p   = r<=80;
    g   = r>80 & r<120;
    m1  = nanmean(fa(g==1, 45:70));
    m2  = nanmean(fa(p==1, 45:70));
    m3  = nanmean(fa(h==1, 45:70));

    % NOTE THAT WE SHOULD CALCULATE SE BASED ON THE NUMBER OF NON NAN SUBS
    se1 = nanstd(fa(g==1, 45:70))./sqrt(size(fa(g==1, 45:70),1));
    se2 = nanstd(fa(p==1, 45:70))./sqrt(size(fa(p==1, 45:70),1));
    se3 = nanstd(fa(h==1, 45:70))./sqrt(size(fa(h==1, 45:70),1));
    
    % plot them
    figure; hold('on');
    patch([1:length(m1) fliplr(1:length(m1))],[m1 - se1, fliplr(m1 + se1)],[.8 0 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
    plot(1:length(m1), m1, '-', 'color', [.8 0 0], 'linewidth',3);
    patch([1:length(m1) fliplr(1:length(m1))],[m2 - se2, fliplr(m2 + se2)],[0 0 .8],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
    plot(1:length(m1), m2, '-', 'color', [0 0 .8], 'linewidth',3);
    patch([1:length(m1) fliplr(1:length(m1))],[m3 - se3, fliplr(m3 + se3)],[0 .8 0],...
        'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 .8 0]);
    plot(1:length(m1), m3, '-', 'color', [0 .8 0], 'linewidth',3);
    
    % plot them out
    figure; hold on;
    plot(fa(g, 57), r(g), 'ko', 'markerfacecolor', [.8 0 0]);lsline
    plot(fa(p, 57), r(p), 'ko', 'markerfacecolor', [0 0 .8]);lsline
    plot(fa(h, 57), r(h), 'ko', 'markerfacecolor', [0 .8 0]);lsline
    xlabel(sprintf('%s (tract num %d) FA', fgnames{fgnum}, fgnum),...
        'fontsize', 14, 'fontname', 'times')
    ylabel('Reading score','fontsize', 14, 'fontname', 'times')
    




%% Let's look at the correlation between different tracts
fgnames = AFQ_get(afq,'fgnames');
valname = 'md';
fa_arcL = nanmean(AFQ_get(afq, 'Left Arcuate', valname),2);
figure;
val = [.6 1]
for ii = 1:20
    subplot(4,5,ii)
    fa(:,ii) = nanmean(AFQ_get(afq, fgnames{ii}, valname),2);
        usesubs = fa_arcL > val(1) & fa(:,ii) > val(1) & fa_arcL < val(2) & fa(:,ii) < val(2)
    plot(fa_arcL(usesubs), fa(usesubs,ii),'ko', 'markerfacecolor', [.5 .5 .5]);
    axis tight
    title(sprintf([fgnames{ii} ' r=%.2f'], corr(fa_arcL(usesubs), fa(usesubs,ii), 'rows', 'pairwise')));
    lsline
end

%% Factor analysis
fa = AFQ_get(afq,'vals','fa');
fa = all(~isnan(fa),2);
[LAMBDA, PSI, T] = factoran(fa,2);





