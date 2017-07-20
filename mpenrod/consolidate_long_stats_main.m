% Script which will read in the longitudinal data from the intervention and
% control groups, consolidating it into a series of tables which can be
% used for other forms of analysis
%
% Author: Mark Penrod
% Date: July 2017

% original file is in this folder
cd('/mnt/scratch/projects/freesurfer')

intervention = {'NLR_102_RS', 'NLR_110_HH', 'NLR_145_AC', 'NLR_150_MG', ...
    'NLR_151_RD', 'NLR_152_TC', 'NLR_160_EK', 'NLR_161_AK', 'NLR_162_EF', ...
    'NLR_163_LF', 'NLR_164_SF', 'NLR_170_GM', 'NLR_172_TH', 'NLR_174_HS', ...
    'NLR_179_GM', 'NLR_180_ZD', 'NLR_199_AM', 'NLR_204_AM', 'NLR_205_AC', ...
    'NLR_206_LM', 'NLR_207_AH', 'NLR_208_LH', 'NLR_210_SB', 'NLR_211_LB'};

control =  {'NLR_101_LG', 'NLR_103_AC', 'NLR_105_BB', 'NLR_127_AM', ...
    'NLR_130_RW', 'NLR_133_ML', 'NLR_146_TF', 'NLR_187_NB', 'NLR_191_DF', ...
    'NLR_195_AW', 'NLR_197_BK', 'NLR_201_GS', 'NLR_202_DD', 'NLR_203_AM', ...
    'RI_124_AT', 'RI_138_LA', 'RI_141_GC', 'RI_143_CH', 'RI_144_OL'};

sub_dir = '/mnt/scratch/projects/freesurfer';
control_outpath  = fullfile(sub_dir,'Control_Data');
inter_outpath = fullfile(sub_dir,'Intervention_Data');

lh_fileBase = 'long.lh.aparc.stats.thickness-';
rh_fileBase = 'long.rh.aparc.stats.thickness-';

lhFOIs = {strcat(lh_fileBase,'avg.dat'),strcat(lh_fileBase,'pc1fit.dat'), ...
    strcat(lh_fileBase,'rate.dat'),strcat(lh_fileBase,'spc.dat')};
rhFOIs = {strcat(rh_fileBase,'avg.dat'),strcat(rh_fileBase,'pc1fit.dat'), ...
    strcat(rh_fileBase,'rate.dat'),strcat(rh_fileBase,'spc.dat')};

meas_labels = {'Measure:thickness-avg','Measure:thickness-pc1fit',...
    'Measure:thickness-rate','Measure:thickness-spc'};

%% Consolidating files

for ii = 1:numel(lhFOIs)
    consolidate_long_stats(sub_dir,control,lhFOIs{ii},meas_labels{ii},control_outpath);
    consolidate_long_stats(sub_dir,control,rhFOIs{ii},meas_labels{ii},control_outpath);
    consolidate_long_stats(sub_dir,intervention,lhFOIs{ii},meas_labels{ii},inter_outpath);
    consolidate_long_stats(sub_dir,intervention,rhFOIs{ii},meas_labels{ii},inter_outpath);
end

