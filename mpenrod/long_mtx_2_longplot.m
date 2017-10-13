% Function which iterates through a 3D matrix and creates a longitudinal
% plot which tracks the change in cortical thickness in the ROIs over time.
%
% Inputs:
% long_mtx: 3D matrix organized such that:
%   Column 1 is the subject ID
%   Columns 2-5 are the thickness average of the ROI at a given timepoint
%   Column 6 is the ROI name
% Each layer of the 3D matrix should denote a separate ROI
% hemi: 'LH' or 'RH'
% covar: the covariate used to identify the ROIs, '' if no covariate used,
% another label if a different means of measurement was used
% sessions: array including which of the 4 sessions that should be plotted
% outpath: folder which the plots will be saved to
%
% Notes on the ouput plots:
% The error bars represent standard error of the mean. Another function
% which can be used to get longitudinal plots is lme_long_fitandplot, which
% uses the linear mixed effects model in order to calculate significance of
% change over time
%
% Author: Mark Penrod
% Date: July 2017


function long_mtx_2_longplot(long_mtx,hemi,covar,sessions,outpath)
mtx_size = size(long_mtx);
for ii = 1:mtx_size(3)
    nn = 1;
    for jj = 1:numel(sessions)
        sess_data = long_mtx(:,1+sessions(jj),ii);
        if iscell(sess_data)
            sess_data = cell2mat(sess_data);
        end
        timepoint_avgs(nn) = mean(sess_data);
        timepoint_SEMs(nn) = std(sess_data)/...
            sqrt(numel(sess_data));
        nn = nn + 1;
    end
    set(0,'DefaultFigureVisible','off');
    fig = figure; hold on
    errorbar(sessions,timepoint_avgs,timepoint_SEMs,'b')
    legend(covar,'Location','southwest');
    box off
    xlabel('Timepoint')
    ylabel(strcat('Cortical Thickness in',[' '],hemi,[' '],long_mtx{1,6,ii}))
    saveas(fig,fullfile(outpath,strcat(hemi,long_mtx{1,6,ii},'_',covar,'.jpg')))
    set(0,'DefaultFigureVisible','on');
end
end