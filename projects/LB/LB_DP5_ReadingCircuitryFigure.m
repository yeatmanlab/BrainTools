% Render fibers for image in dp5 grant
t1Path = '/home/jyeatman/projects/Lindamood_Bell/MRI/anatomy/LB7/t1.nii.gz'
t1 = readFileNifti(t1Path);
%t1.data = mrAnatHistogramClip(t1.data,.1,.99)
fdir = '/home/jyeatman/projects/Lindamood_Bell/MRI/child/control/session2/LB7_20130802/20130802_1028/dti80trilin_mrtrix/fibers/DP5figs'
fgNames = {'L_Arcuate_VOT_clean.mat' 'L_postArcuate.mat' 'VOF_VWFA.mat' 'Left_LIF_VOT.mat' 'L_OpticRad_clean.mat'}
%colors = [.8 0 0; 1 .5 0;0 .8 0;1 1 0;0 0 .8];
colors = jet(5)
nfibers = 100;
for ii = 1:length(fgNames)
   fg(ii) = dtiReadFibers(fullfile(fdir,fgNames{ii}))
end

lh=AFQ_RenderFibers(fg(1),'numfibers',nfibers,'color',colors(1,:));axis off
for ii = 2:length(fgNames)
    AFQ_RenderFibers(fg(ii),'numfibers',nfibers,'color',colors(ii,:),'newfig',0)
end
AFQ_AddImageTo3dPlot(t1,[-12 0 0]);
s=AFQ_AddImageTo3dPlot(t1,[0 0 -20]);
view(300,20)
view(-55,25)

camlight(lh,'right')
lgnh = AFQ_RenderEllipsoid(eye(3).*50,[-20 -15 -2],50,[.7 0 0],0)
vwfah = AFQ_RenderEllipsoid(eye(3).*120,[-45 -62 -17],50,[.7 0 0],0)
set(vwfah,'facealpha',.6)
print('-dpng','-r300','ReadingCircuitry.png')