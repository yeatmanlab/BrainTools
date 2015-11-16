function mri_2dhist(im1,im2,drange,cax)

   
figure;hold;
mask = im1 > min(drange) & im2 > min(drange) & im1 < max(drange) & im2 < max(drange);
hist2d(im1(mask),im2(mask),drange, drange);
colorbar;
if exist('cax','var') && ~isempty(cax)
    caxis(cax);
end
axis image
plot([min(drange) max(drange)],[min(drange) max(drange)],'--w','linewidth',2);
R2 = calccod(im1(mask),im2(mask));
r = corr(im1(mask),im2(mask));
fprintf('\nR^2 = %.2f    r = %.2f',R2,r);
dimg = im1-im2;
dimg(~mask) = 0;
showMontage(dimg); colormap jet

xi = min(drange):1:max(drange);
f1 = ksdensity(im1(mask),xi);
f2 = ksdensity(im2(mask),xi);
figure; hold;
plot(xi,f1,'-r');
plot(xi,f2,'-b');