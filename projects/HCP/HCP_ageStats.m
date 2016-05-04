% Collection of tools to organize data by age and run statistical analyses


%% Group ages

ageRng = unique(afq.metadata.Age);
afq.age = zeros(numel(afq.sub_names),1);

for ii = 1:numel(afq.sub_names)
    if  strcmp('22-25', cell(afq.metadata.Age(ii)))
        afq.age(ii) = 22;
    elseif strcmp('26-30', cell(afq.metadata.Age(ii)))
        afq.age(ii) = 26;
    elseif strcmp('31-35', cell(afq.metadata.Age(ii)))
        afq.age(ii) = 31;
    elseif strcmp('36+', cell(afq.metadata.Age(ii)))
        afq.age(ii) = 36;
    end
end
%% Compare arcuate FA means

af22 = nanmean(afq.vals.fa{19}(afq.age == 22, :),2);
af26 = nanmean(afq.vals.fa{19}(afq.age == 26, :),2);
af31 = nanmean(afq.vals.fa{19}(afq.age == 31, :),2);
af36 = nanmean(afq.vals.fa{19}(afq.age == 36, :),2);

af22m = nanmean(af22);
af22sd = nanstd(af22);
af26m = nanmean(af26);
af26sd = nanstd(af26);
af31m = nanmean(af31);
af31sd = nanstd(af31);
af36m = nanmean(af36);
af36sd = nanstd(af36);

% Flat

%% 

% Scatter plots
agebins = [24 28 32];
for fgnum = 1:20;
    fgnames = afq.fgnames{fgnum};
    
    fa22 = nanmean(afq.vals.fa{fgnum}(afq.age == 22, :),2);
    fa26 = nanmean(afq.vals.fa{fgnum}(afq.age == 26, :),2);
    fa31 = nanmean(afq.vals.fa{fgnum}(afq.age == 31, :),2);
    faall = nanmean(afq.vals.fa{fgnum}(afq.age == 22 | afq.age == 26 | afq.age == 31, :), 2);
    %     fa36 = nanmean(afq.vals.fa{fgnum}(afq.age == 36, :),2);
    
    md22 = nanmean(afq.vals.md{fgnum}(afq.age == 22, :),2);
    md26 = nanmean(afq.vals.md{fgnum}(afq.age == 26, :),2);
    md31 = nanmean(afq.vals.md{fgnum}(afq.age == 31, :),2);
    mdall = nanmean(afq.vals.md{fgnum}(afq.age == 22 | afq.age == 26 | afq.age == 31, :), 2);
    %     md36 = nanmean(afq.vals.fa{fgnum}(afq.age == 36, :),2);
    
    % Concatenate means
    
    fa = vertcat(nanmean(fa22), nanmean(fa26), nanmean(fa31));
    md = vertcat(nanmean(md22), nanmean(md26), nanmean(md31));
    
    % Concatenate standard errors
    sefa = vertcat(nanstd(fa22)./sqrt(numel(fa22)), nanstd(fa26)./sqrt(numel(fa26)), nanstd(fa31)./sqrt(numel(fa31)));
    semd = vertcat(nanstd(md22)./sqrt(numel(md22)), nanstd(md26)./sqrt(numel(md26)), nanstd(md31)./sqrt(numel(md31)));
    
    % plot them out
    figure(1);
    subplot(4,5,fgnum); hold on;
    errorbar(agebins, fa, 2.*sefa, 'ko', 'markerfacecolor', [.8 0 0]);%lsline
    plot(agebins, repmat(nanmean(faall),1,3), '--k');
    xlabel('Age Bins', 'fontsize', 14, 'fontname', 'times');
    ylabel(sprintf('%s (tract num %d) FA', fgnames, fgnum),'fontsize', 14, 'fontname', 'times')
    title(fgnames)
    %axis([1 3 .39 .64])
    axis('tight')
    set(gca,'xlim',[20 35])
    
    figure(2)
    subplot(4,5,fgnum); hold on;
    errorbar(agebins, md, 2.*semd, 'ko', 'markerfacecolor', [0 0 .8]);%lsline
    plot(agebins, repmat(nanmean(mdall),1,3), '--k');
    xlabel('Age Bins', 'fontsize', 14, 'fontname', 'times');
    ylabel(sprintf('%s (tract num %d) MD', fgnames, fgnum),'fontsize', 14, 'fontname', 'times')
    title(fgnames)
    %axis([1 3 .72 .81])
    axis('tight')
        set(gca,'xlim',[20 35])

%     % Average tract profiles
%     g   = r>85;
%     m1  = nanmean(fa(g==1, :));
%     m2  = nanmean(fa(g==0, :));
%     % NOTE THAT WE SHOULD CALCULATE SE BASED ON THE NUMBER OF NON NAN SUBS
%     se1 = nanstd(fa(g==1, :))./sqrt(size(fa(g==1, :),1));
%     se2 = nanstd(fa(g==0, :))./sqrt(size(fa(g==0, :),1));
%     % plot them
%     figure; hold('on');
%     patch([1:100 fliplr(1:100)],[m1 - se1, fliplr(m1 + se1)],[.8 0 0],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [.8 0 0]);
%     h(1) = plot(1:100, m1, '-', 'color', [.8 0 0], 'linewidth',3);
%     title(fgnames(fgnum));
%     patch([1:100 fliplr(1:100)],[m2 - se2, fliplr(m2 + se2)],[0 0 .8],...
%         'facealpha',.5, 'edgealpha', .8, 'edgecolor', [0 0 .8]);
%     h(2) =  plot(1:100, m2, '-', 'color', [0 0 .8], 'linewidth',3);
%     title(fgnames(fgnum));
%     legend(h,'Good readers', 'Poor readers');
    
end