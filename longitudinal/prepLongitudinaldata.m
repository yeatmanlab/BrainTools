


% Select group of Subjects
% read data from Desktop
[tmp, ~, data] = xlsread('~/Desktop/NLR_Scores.xlsx');
% gather column headings
data_ref = data(1,:);
% remove data headers from data
data = data(2:end,:);
% create array of subjects of interest
subs = {'201_GS', '202_DD', '203_AM', '204_AM', '205_AC', '206_LM'};
% find all rows for subjects of interest
data_indx_tmp = [];
data_indx     = [];
for ii = 1:numel(subs)
    data_indx_tmp = find(strcmp(data(:, strcmp(data_ref, 'Subject')), subs(ii)));
    data_indx = vertcat(data_indx, data_indx_tmp);
end
% create refined data array for data of interest
for ii = 1:numel(data_indx)
    sid(ii)         = data(data_indx(ii), strcmp(data_ref, 'Subject'));
    sessnum(ii)     = data(data_indx(ii), strcmp(data_ref, 'Visit'));
    time(ii)        = data(data_indx(ii), strcmp(data_ref, 'Time'));
    hours(ii)       = data(data_indx(ii), strcmp(data_ref, 'Hours'));
    % READING TESTS
    wj_brs(ii)      = data(data_indx(ii), strcmp(data_ref, 'WJ_BRS'));
    wj_rf(ii)       = data(data_indx(ii), strcmp(data_ref, 'WJ_RF'));
    wj_lwid(ii)     = data(data_indx(ii), strcmp(data_ref, 'WJ_LWID_SS'));
    wj_wa(ii)       = data(data_indx(ii), strcmp(data_ref, 'WJ_WA_SS'));
    twre_swe(ii)    = data(data_indx(ii), strcmp(data_ref, 'TOWRE_SWE_SS'));
    twre_pde(ii)    = data(data_indx(ii), strcmp(data_ref, 'TOWRE_PDE_SS'));
    twre_indx(ii)   = data(data_indx(ii), strcmp(data_ref, 'TWRE_INDEX'));
    wasi(ii)        = data(data_indx(ii), strcmp(data_ref, 'WASI_FS2'));
    elision(ii)     = data(data_indx(ii), strcmp(data_ref, 'CTOPP_ELISION_SS'));
    ctopp_pa(ii)    = data(data_indx(ii), strcmp(data_ref, 'CTOPP_PA'));
    ctopp_rn(ii)    = data(data_indx(ii), strcmp(data_ref, 'CTOPP_RAPID'));
    % WORD LISTS
    wl_4let(ii)     = data(data_indx(ii), strcmp(data_ref, 'WL_4let'));
    wl_5let(ii)     = data(data_indx(ii), strcmp(data_ref, 'WL_5let'));
end

% create dataset variable condensing the information

dataset = dataset(sid, sessnum, time, hours, wj_lwid, wj_wa, wj_brs, ...
    wj_rf, twre_swe, twre_pde, twre_indx, wasi, elision, ctopp_pa, ...
    ctopp_rn);








