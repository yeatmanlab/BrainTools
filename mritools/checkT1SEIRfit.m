function checkT1SEIRfit(inputDir)
% Check how the T1 curves fit the raw SEIR data after running mrQ
%
% checkT1SEIRfit(inputDir)
%

%% load data
if ~exist('inputDir') || isempty(inputDir)
    inputDir='/mnt/diskArray/projects/DISC/20160329_T1_TSEIR_Test_Human/mrQ/Phil'; % GE folde or Phillips folder
end
load(fullfile(inputDir,'SEIR_epi_1/data/SEIR_Dat.mat')); %% the aligned raw data, loads 'data' xform' 'extra'
load(fullfile(inputDir,'SEIR_epi_1/fitT1_GS/T1FitNLSPR_SEIR_Dat.mat')) %% the data fit parameters: loads: 'll_T1' 'nlsS'

% ll_T1 has the four parameters of the T1 fit for each voxel, namely:
% (1) T1
% (2) 'b' parameter from the model
% (3) 'a' parameter
% (4) residual from the fit

%% raw data organization
data = abs(data);
dims  = size(data);
nbrow = size(data,1);
nbcol = size(data,2);

if numel(dims) > 3
    nbslice = dims(3); % Check number of slices
else
    nbslice = 1;
    tmpData(:,:,1,:) = data; % Make data a 4-D array regardless of number of slices
    data = tmpData;
    clear tmpData;
end

%% Data check: Fit inspection

% Quick check of the fit
TI    = extra.tVec;
nbtp  = 20;
timef = linspace(min(TI),max(TI),nbtp);

% Inserting a short pause, otherwise some computers seem to have problems
pause(1);

zz = 0;
while(true)
    
    disp('For which slice would you like to check the fit?');
    zz = input(['Enter number 1 to ' num2str(nbslice) '.  0 for no check --- '], 's');
    zz = cast(str2double(zz), 'int16');
    if (isinteger(zz) && zz >= 0 && zz <= nbslice)
        break;
    end
end

if (zz ~= 0)
    sliceData = squeeze(data(:,:,zz,:));
    datafit = zeros(nbrow,nbcol,nbtp);
    
    
    for kk = 1:20
        datafit(:,:,kk) = abs(ll_T1(:,:,zz,3) + ...
            ll_T1(:,:,zz,2).*exp(-timef(kk)./ll_T1(:,:,zz,1)));
    end
    
    
    % Check Data: Fit inspection
    disp('Click on one point to check the fit. CTRL-click or right-click when done')
    plotData(real(sliceData),TI,real(datafit),squeeze(ll_T1(:,:,zz,1)));
    close all
end

