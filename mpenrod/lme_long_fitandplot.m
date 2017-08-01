%
% Function which takes data of subjects across timepoints, identified from
% a correlation with a given covariate and produced a plot tracking the
% change in cortical thickness of the ROI across time. The linear mixed
% effects model is used to determine the statistical significance of these
% changes.
%
% Inputs:
% long_mtx: A matrix organized such that column 1 is subject ID, column 2
% is a single average thickness value for the ROI, and column 3 is the
% session number/timepoint
% ROIs: A list of ROI labels
% hemi: the hemisphere of interest ('LH' or 'RH')
% covar: the covariate used to identify the correlational ROIs. If the ROIs
% did not come from a correlational analysis, an alternative label can be
% used
% outpath: the path to the folder where the graphs will be saved
%
% Ouput:
% A series of plots which will show the change in cortical thickness across
% timepoints 
%
% Author: Mark Penrod
% Date: July, 2017

function lme = lme_long_fitandplot(long_mtx,ROIs,hemi,covar,outpath)
%% fit longitudinal model
dataIn = long_mtx;

% temp replace blanks with NaN
for ii = 1:numel(dataIn(:,2))
    if isempty(dataIn{ii,2})
        dataIn{ii,2} = NaN;
    end
end
for ii = 1:numel(ROIs)
    roiName = ROIs{ii};
    roiIndx = find(strcmp(dataIn(:,4), roiName));
    
    dtable = table(cat(1,dataIn{roiIndx,1}), cat(1,dataIn{roiIndx,2}), categorical(cat(1,dataIn{roiIndx,3})),...
        'VariableNames', {'Sub', 'CT', 'Session'});
    
    % predict cortical thickness, CT, from session, with a random effect of
    % subject:
    lme = fitlme(dtable, 'CT ~ Session + (1|Sub)');
    
    pValue = lme.anova.pValue(2);
    intercept = lme.Coefficients.Estimate(1);
    CT_sess2 = lme.Coefficients.Estimate(1) + lme.Coefficients.Estimate(2);
    CT_sess3 = lme.Coefficients.Estimate(1) + lme.Coefficients.Estimate(3);
    CT_sess4 = lme.Coefficients.Estimate(1) + lme.Coefficients.Estimate(4);
    error_sess1 = lme.Coefficients.SE(1);
    error_sess2 = lme.Coefficients.SE(2);
    error_sess3 = lme.Coefficients.SE(3);
    error_sess4 = lme.Coefficients.SE(4);
    
    fig = plot(1:4, [intercept, CT_sess2, CT_sess3, CT_sess4],'.k', 'MarkerSize', 20);
    hold on
    errorbar(1:4, [intercept, CT_sess2, CT_sess3, CT_sess4], [error_sess1,error_sess2,error_sess3,error_sess4],...
        'k', 'LineStyle', 'none')
    xlabel('Session');
    ylabel(strcat('Cortical Thickness in ',[' ',roiName(1:strfind(roiName,'_')-1)]));
    title(strcat('p-value = ',[' ',num2str(pValue)]));
    hold off
    saveas(fig,fullfile(outpath,strcat('lme_long_',covar,'_',hemi,roiName,'.jpg')));
end
end