% Title: Create Correlation Matrix, Reality & Timing
% Source: MathWorks, RE: corrplot()
% Author: Patrick Donnelly, BDE Lab, U of Washington
% Date: October 21, 2015

% Read XLS file
% xlsread('Spreadsheet_name.xlsx', 'Page_name');
[~,~,data] = xlsread('/mnt/diskArray/projects/NLR/NLR_Scores.xlsx', 'Dyslexic', 'A1:AS36');

% Concatenate data into column vectors
dl_wid         = vertcat(data{2:end,5});
dl_wa          = vertcat(data{2:end,7});
dl_or          = vertcat(data{2:end,9});
dl_srf         = vertcat(data{2:end,11});
dl_wj_brs      = vertcat(data{2:end,12});
dl_wj_rf       = vertcat(data{2:end,13});
dl_twre        = vertcat(data{2:end,19});
dl_ctopp_ran   = vertcat(data{2:end,43});
dl_ctopp_elds  = vertcat(data{2:end,45});

X(:,1) = dl_wj_brs;
X(:,2) = dl_wj_rf;
X(:,3) = dl_twre;
X(:,4) = dl_ctopp_elds;
X(:,5) = dl_ctopp_ran;
%

% Create Correlation Matrix
dbstop if error
addpath('/usr/local/MATLAB/R2014a/toolbox/matlab/scribe/');
[R,PValue] = corrplot(X, 'type', 'Kendall', 'testR', 'on','varNames', {'BRS', 'RF', 'TWRE', 'ELDS', 'RAN'});