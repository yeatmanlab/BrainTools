function png2avi(pngdir,filestr, outname)

if ~exist('filestr','var') || isempty(filestr)
    filestr = '*.png';
end
d = dir(fullfile(pngdir,filestr));
for ii = 1:length(d)
   im(:,:,:,ii) = imread(fullfile(pngdir,d(ii).name));
end
v = VideoWriter(outname);
open(v);
writeVideo(v,im);
close(v);