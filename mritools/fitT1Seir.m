function [T1image, Cimage, Gimage] = fitT1Seir(niiPaths,TI,slices)
% Fit T1 from spin-echo inversion-recovery data
% 
% [T1image, Cimage, Gimage] = fitT1Seir(niiPaths,TI,[slices])
%
% Example:
%
% niiPaths = {'seir80_real_epi0001.nii.gz', 'seir200_real_epi0003.nii.gz'...
% , 'seir400_real_epi0005.nii.gz', 'seir1200_real_epi0007.nii.gz',...
% 'seir2400_real_epi0009.nii.gz'};
% TI = [80 200 400 1200 2400];
%
% [T1image, Cimage, Gimage] = fitT1Seir(niiPaths,TI);
%


% Load each volume
for ii = 1:length(niiPaths)
    im(ii) = readFileNifti(niiPaths{ii});
    mask(:,:,:,ii) = abs(smooth3(abs(smooth3(im(ii).data,'gaussian',9))>10,'gaussian',9))>.1;
    % and covert volume data to a vector
    vdata(ii,:) = double(im(ii).data(:));
end
mask = max(mask,[],4);
% Find linear indices of voxels to analyze
vmask = find(mask(:));

% Options for curve fitting algorithm
options = optimoptions(@lsqcurvefit,'Algorithm','levenberg-marquardt', 'display','off');

%% Fit T1 in each voxel

% Break into steps so we can more easily monitor progress
nslice = size(im(1).data,3);
voxslice = prod(size(im(1).data(:,:,1)));
step = [0:nslice-1].*voxslice+1;
% Select which slices to fit
if ~exist('slices','var') || isempty(slices)
    slices = 1:nslice;
end
c=0;
tic;
% Preallocate a matrix of parameters. 3 Parameters per voxel
p = zeros(3,voxslice*max(slices));
for s = step(slices)
    fprintf('Fitting voxel %d. Time ellapsed = %.2fs\n',s,toc)
    c = c+1;
    % Voxels within this slice
    vx = s:s+voxslice-1;
    % Of the voxels within the slice which are within the mask
    vx = vx(ismember(vx,vmask));
    % This will be the starting point for the fitting
    strt = [100 1 1000];
    for v = vx
        
        % Fit T1
        %p(:,v) = lsqcurvefit(@(p,TI) IRfunction(p,TI),strt,TI,vdata(:,v)',[],[],options);
        p(:,v) = lsqcurvefit(@(p,TI) IRfunction(p,TI),[mean(vdata(:,v)) 1 1000],TI,vdata(:,v)',[],[],options);
        
    end
    
end

% Put values back into image
T1image = reshape(p(3,:),size(im(1).data(:,:,1:max(slices))));
Cimage = reshape(p(1,:),size(im(1).data(:,:,1:max(slices))));
Gimage = reshape(p(2,:),size(im(1).data(:,:,1:max(slices))));

T1image(T1image<0) = 0; T1image(T1image>5000) = 5000;
showMontage(T1image); colorbar;

save T1fit

% Display
viewSEIRfit(T1image, Cimage, Gimage, niiPaths, TI);

return

function s = IRfunction(params,TI,mag)

if exist('mag','var') && ~isempty(mag) && mag == 1
    % |C + G*exp(-TI/T1)|
    s = abs(params(1) + params(2)*exp(-TI./params(3)));
else
    % C + G*exp(-TI/T1)
    s = params(1) + params(2)*exp(-TI./params(3));
end

return

function t1GridSeach(T1vals,TI)

if ~exist('T1vals','mag') || isempty(T1vals)
    T1vals = 500:10:5000
end

for ii = 1:length(TI)
    expvals(ii,:) = exp(-(TI(ii))./T1vals);
end

return
