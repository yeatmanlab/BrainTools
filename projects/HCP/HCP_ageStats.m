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
for fgnum = 1:20;
    fgnames = afq.fgnames{fgnum};
    
    fa22 = nanmean(afq.vals.fa{fgnum}(afq.age == 22, :),2);
    fa26 = nanmean(afq.vals.fa{fgnum}(afq.age == 26, :),2);
    fa31 = nanmean(afq.vals.fa{fgnum}(afq.age == 31, :),2);
%     fa36 = nanmean(afq.vals.fa{fgnum}(afq.age == 36, :),2);
    
    md22 = nanmean(afq.vals.md{fgnum}(afq.age == 22, :),2);
    md26 = nanmean(afq.vals.md{fgnum}(afq.age == 26, :),2);
    md31 = nanmean(afq.vals.md{fgnum}(afq.age == 31, :),2);
%     md36 = nanmean(afq.vals.fa{fgnum}(afq.age == 36, :),2);

% Concatenate means

    fa = vertcat(nanmean(fa22), nanmean(fa26), nanmean(fa31));
    md = vertcat(nanmean(md22), nanmean(md26), nanmean(md31));

    % plot them out
    figure;
    subplot(1,2,1); hold on;
    plot(fa, 'ko', 'markerfacecolor', [.8 0 0]);lsline
    xlabel('Age Bins', 'fontsize', 14, 'fontname', 'times');
    ylabel(sprintf('%s (tract num %d) FA', fgnames, fgnum),'fontsize', 14, 'fontname', 'times')
    
    subplot(1,2,2); hold on;
    plot(md, 'ko', 'markerfacecolor', [0 0 .8]);lsline
    xlabel('Age Bins', 'fontsize', 14, 'fontname', 'times');
    ylabel(sprintf('%s (tract num %d) MD', fgnames, fgnum),'fontsize', 14, 'fontname', 'times')
    
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