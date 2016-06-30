% Title: Create Correlation Matrix
% Source: MathWorks, RE: corrplot()
% Author: Patrick Donnelly, BDE Lab, U of Washington
% Date: October 21, 2015

%% Get dataset
[~, ~, data] = xlsread('~/Desktop/NLR_Scores.xlsx');
%[~, ~, data] = xlsread('C:\Users\Patrick\Desktop/NLR_Scores.xlsx');

%% Select group of Subjects
subs = {'152_TC', '201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM'};
% gather column headings
data_ref = data(1,:);
% add '\' preceding each "_" for nicer looking titles/formatting
data_ref = strrep(data_ref, '_', '\_');
% remove data headers from data
data = data(2:end,:);
% find all rows for subjects of interest
data_indx_tmp = [];
data_indx     = [];
for subj = 1:numel(subs)
    data_indx_tmp = find(strcmp(data(:, strcmp(data_ref, 'Subject')), subs(subj)));
    data_indx = vertcat(data_indx, data_indx_tmp);
end
% create refined data array for data of interest
% initialize empty arrays
lwid = []; wa = []; brs = []; rf = []; swe = []; pde = []; twre = []; wasi = []; elision = []; pa = []; rapid = [];
% vertcat each reading test variable
for subj = 1:numel(data_indx)
    lwid       = vertcat(lwid, data(data_indx(subj), strcmp(data_ref, 'WJ\_LWID\_SS')));
    wa         = vertcat(wa, data(data_indx(subj), strcmp(data_ref, 'WJ\_WA\_SS')));
    brs        = vertcat(brs, data(data_indx(subj), strcmp(data_ref, 'WJ\_BRS')));
    rf         = vertcat(rf, data(data_indx(subj), strcmp(data_ref, 'WJ\_RF')));
    swe        = vertcat(swe, data(data_indx(subj), strcmp(data_ref, 'TWRE\_SWE\_SS')));
    pde        = vertcat(pde, data(data_indx(subj), strcmp(data_ref, 'TWRE\_PDE\_SS')));
    twre       = vertcat(twre, data(data_indx(subj), strcmp(data_ref, 'TWRE\_INDEX')));
    wasi       = vertcat(wasi, data(data_indx(subj), strcmp(data_ref, 'WASI\_FS2')));
    elision    = vertcat(elision, data(data_indx(subj), strcmp(data_ref, 'CTOPP\_ELISION\_SS')));
    pa         = vertcat(pa, data(data_indx(subj), strcmp(data_ref, 'CTOPP\_PA')));
    rapid      = vertcat(rapid, data(data_indx(subj), strcmp(data_ref, 'CTOPP\_RAPID')));
end

%% Concatenate data into column vectors
% dl_wid         = vertcat(data{2:end,5});
% dl_wa          = vertcat(data{2:end,7});
% dl_or          = vertcat(data{2:end,9});
% dl_srf         = vertcat(data{2:end,11});
% dl_wj_brs      = vertcat(data{2:end,12});
% dl_wj_rf       = vertcat(data{2:end,13});
% dl_twre        = vertcat(data{2:end,19});
% dl_ctopp_ran   = vertcat(data{2:end,43});
% dl_ctopp_elds  = vertcat(data{2:end,45});
% 
% X(:,1) = dl_wj_brs;
% X(:,2) = dl_wj_rf;
% X(:,3) = dl_twre;
% X(:,4) = dl_ctopp_elds;
% X(:,5) = dl_ctopp_ran;
%

% Create Correlation Matrix
dbstop if error
addpath('/usr/local/MATLAB/R2014a/toolbox/matlab/scribe/');
[R,PValue] = corrplot(X, 'type', 'Kendall', 'testR', 'on','varNames', {'BRS', 'RF', 'TWRE', 'ELDS', 'RAN'});