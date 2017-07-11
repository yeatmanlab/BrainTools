% Makes and runs the aparcstatstable command
interSubID = {'NLR_145_AC', 'NLR_151_RD', 'NLR_161_AK', 'NLR_172_TH',...
    'NLR_180_ZD', 'NLR_208_LH', 'NLR_102_RS', 'NLR_150_MG', 'NLR_152_TC', ...
    'NLR_162_EF', 'NLR_174_HS', 'NLR_210_SB', 'NLR_110_HH', 'NLR_160_EK', ...
    'NLR_170_GM', 'NLR_179_GM', 'NLR_207_AH', 'NLR_211_LB', 'NLR_164_SF', ...
    'NLR_204_AM', 'NLR_206_LM', 'NLR_163_LF', 'NLR_205_AC', 'NLR_199_AM'};
freesurf_dir = '/mnt/scratch/projects/freesurfer';
cd(freesurf_dir)
lh_long_cmd = 'aparcstats2table --subjects ';
rh_long_cmd = lh_long_cmd;
for ii = 1:numel(interSubID)
    subject = interSubID{ii};
    for ss = 1:4
        if exist (fullfile(freesurf_dir,strcat(subject,'_',num2str(ss),...
                '.long.',subject,'_template')),'file')
            lh_long_cmd = strcat(lh_long_cmd,[' ',subject],'_',num2str(ss),...
                '.long.',subject,'_template');
            rh_long_cmd = strcat(rh_long_cmd,[' ',subject],'_',num2str(ss),...
                '.long.',subject,'_template');
        end
    end
end
lh_long_cmd = strcat(lh_long_cmd,' --hemi lh --meas thickness --tablefile lh_long_aparc_stats.txt');
rh_long_cmd = strcat(rh_long_cmd,' --hemi rh --meas thickness --tablefile rh_long_aparc_stats.txt');
system(lh_long_cmd)
system(rh_long_cmd)
