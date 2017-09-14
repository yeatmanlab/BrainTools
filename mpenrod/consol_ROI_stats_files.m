
% Function which takes a series of stats files created by
% mris_anatomicalstats and consolidates them into a single file
%
% Inputs:
% stats_folder: The path to the folder containing the individual files
% line_of_interest: The line on the file where the data is located
%
% Output:
% ROI_table: A single table containing all data from the stats files,
% organized in the same manner. The table can either be manipulated and
% analyzed in MATLAB or written to a file.
%
% Author: Mark Penrod
% Date: July 2017


function ROI_table = consol_ROI_stats_files(stats_folder,line_of_interest)
stats_files = dir(fullfile(stats_folder,'*.stats'));


for ii = 1:numel(stats_files)
    file = importdata(fullfile(stats_folder,stats_files(ii).name),'\t',line_of_interest);
    file_data = strsplit(char(file(line_of_interest)));
    file_data = file_data(2:end);
    data{ii,1} = stats_files(ii).name;
    for jj = 1:numel(file_data)
        data{ii,jj+1} = file_data{jj};
    end
end
variable_names = {'subid' 'NumVert' 'SurfArea' 'GrayVol' 'ThickAvg',...
    'ThickStd' 'MeanCurv' 'GausCurv' 'FoldInd' 'CurvInd'};
for ii = 1:cols(data)
    struct.(variable_names{ii}) = data(:,ii);
end
ROI_table = struct2table(struct);
end
