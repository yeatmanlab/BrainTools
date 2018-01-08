% Makes the aparcstatstable command
interSubID = {'NLR_145_AC', 'NLR_151_RD', 'NLR_161_AK', 'NLR_172_TH',...
    'NLR_180_ZD', 'NLR_208_LH', 'NLR_102_RS', 'NLR_150_MG', 'NLR_152_TC', ...
    'NLR_162_EF', 'NLR_174_HS', 'NLR_210_SB', 'NLR_110_HH', 'NLR_160_EK', ...
    'NLR_170_GM', 'NLR_179_GM', 'NLR_207_AH', 'NLR_211_LB', 'NLR_164_SF', ...
    'NLR_204_AM', 'NLR_206_LM', 'NLR_163_LF', 'NLR_205_AC', 'NLR_199_AM'};
freesurf_dir = '/mnt/scratch/projects/freesurfer';
cd(freesurf_dir)
lh_basic_cmd = 'aparcstats2table --subjects ';
lh_long_cmd = lh_basic_cmd;
rh_basic_cmd = lh_basic_cmd;
rh_long_cmd = lh_basic_cmd;
for ii = 1:numel(interSubID)
    subject = interSubID{ii};
    for ss = 1:4
        if exist(fullfile(freesurf_dir,strcat(subject,'_',num2str(ss))),'file')
            lh_basic_cmd = strcat(lh_basic_cmd,[' ',subject],'_',num2str(ss));
            rh_basic_cmd = strcat(rh_basic_cmd,[' ',subject],'_',num2str(ss));
        end
        if exist (fullfile(freesurf_dir,strcat(subject,'_',num2str(ss),...
                '.long.',subject,'_template')),'file')
            lh_long_cmd = strcat(lh_long_cmd,[' ',subject],'_',num2str(ss),...
                '.long.',subject,'_template');
            rh_long_cmd = strcat(rh_long_cmd,[' ',subject],'_',num2str(ss),...
                '.long.',subject,'_template');
        end
    end
end
lh_basic_cmd = strcat(lh_basic_cmd,' --hemi lh --meas thickness --tablefile lh_basic_aparc_stats.txt');
rh_basic_cmd = strcat(rh_basic_cmd,' --hemi rh --meas thickness --tablefile rh_basic_aparc_stats.txt');
lh_long_cmd = strcat(lh_long_cmd,' --hemi lh --meas thickness --tablefile lh_long_aparc_stats.txt');
rh_long_cmd = strcat(rh_long_cmd,' --hemi rh --meas thickness --tablefile rh_long_aparc_stats.txt');
system(lh_basic_cmd)
system(rh_basic_cmd)
system(lh_long_cmd)
system(rh_long_cmd)
