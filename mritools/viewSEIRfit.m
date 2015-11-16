function viewSEIRfit(T1image, Cimage, Gimage, nii, TI)
% Visualize T1 fits to spin-echo inversion-recovery data
%
% viewSEIRfit(T1image, Cimage, Gimage, nii, TI)
%
% Example:
%
% load Users/jyeatman/Research/ILABS/seir_epi_sorted/T1fit.mat
% viewSEIRfit(T1image, Cimage, Gimage, niiPaths, TI)

if ~exist('TI','var') || isempty(TI)
    TI=5000;
end
% Load up image data
if iscell(nii) && ischar(nii{1})
    for ii = 1:length(nii)
        im = readFileNifti(nii{ii});
        imdata(:,:,:,ii) = im.data;
    end
else
    for ii = 1:length(nii)
        imdata(:,:,:,ii) = nii(ii).data;
    end
end

z = input('\nWhich slice number would you like to see?');

% Show image of the slice
figure(1);
imagesc(squeeze(T1image(:,:,z)));colormap gray; colorbar;
title('Choose a point to plot (return to exit)','fontsize',20);
con = 1;
while con == 1
    % get input
    figure(1);
    [y,x] = ginput(1);
    
    % If a point was chosen then plot it. Otherwise exit
    if ~isempty(x) || ~isempty(y)
        x = round(x); y = round(y);
        
        % Plot IR curve
        if exist('TI','var') && ~isempty(TI)
            xx = 0:10:1.3*max(TI);
        else
            xx = 0:10:5000;
        end
        t1 = T1image(x,y,z);
        c = Cimage(x,y,z);
        g = Gimage(x,y,z);
        IRcurve = IRfunction([c g t1],xx);
        
        figure; hold;
        plot(xx,IRcurve);
        
        % Plot IR data
        if exist('imdata','var') && ~isempty(imdata)
            % vector of data points
            IRdata = squeeze(imdata(x,y,z,:));
            plot(TI,IRdata,'ko');
        end
        title(sprintf('(%d,%d,%d) T1 = %dms',x,y,z,round(t1)));
    else
        con = 0;
    end
end
return

function signal = IRfunction(params,TI)
% C + G*exp(-TI/T1)

signal = params(1) + params(2)*exp(-TI./params(3));