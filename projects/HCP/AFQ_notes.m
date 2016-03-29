fg = fgRead('MoriGroups_clean_D5_L4.mat')

c = hsv(20)
for ii = 1:20
   AFQ_RenderFibers(fg(ii), 'numfibers', 100, 'color',c(ii,:));
    
end

% Render the arcuate for each subject
for ii = 1:length(afq.files.fibers.clean)
    fg = fgRead(afq.files.fibers.clean{ii});
    AFQ_RenderFibers(fg(19), 'numfibers', 100, 'color',c(ii,:))
end
% Plot out fa values for the arcuate for each subject
plot(afq.vals.fa{19}')

%% For example to look at correlations between behavioral measures with the arcuate
corr(behavioraldata,afq.vals.fa{19}')

%%
afq=AFQ_Create('sub_dirs',sub_dirs,'sub_group',1,'seedVoxelOffsets',.5)