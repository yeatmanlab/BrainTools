cd /mnt/diskArray/projects/VOF/test/sni-storage/kalanit/biac2/kgs/projects/Longitudinal/Diffusion/JG24/96dir_run1
fg_stt = dtiReadFibers('dti96trilin_stt/fibers/MoriGroups_clean_D5_L4.mat');
fg_mrt = dtiReadFibers('dti96trilin/fibers/MoriGroups_clean_D5_L4.mat');

lh = AFQ_RenderFibers(fg_mrt(19),'numfibers',300,'color',[1 .3 .1],'radius',.3)
AFQ_RenderFibers(fg_mrt(3),'numfibers',300,'color',[1 .3 .1],'radius',.3,'newfig',0)
axis image off
print('-dpng','-r200','~/arcuate+cst_mrt_lat.png')
view(0,0);camlight(lh)
print('-dpng','-r200','~/arcuate+cst_mrt_post.png')

lh = AFQ_RenderFibers(fg_stt(19),'numfibers',300,'color',[.1 .3 1],'radius',.3)
AFQ_RenderFibers(fg_stt(3),'numfibers',300,'color',[.1 .3 1],'radius',.3,'newfig',0)
axis image off
print('-dpng','-r200','~/arcuate+cst_stt_lat.png')
view(0,0);camlight(lh)
print('-dpng','-r200','~/arcuate+cst_stt_post.png')

lh = AFQ_RenderFibers(fg_mrt(19),'numfibers',300,'color',[1 .3 .1],'radius',.3)
AFQ_RenderFibers(fg_mrt(3),'numfibers',300,'color',[1 .3 .1],'radius',.3,'newfig',0)
AFQ_RenderFibers(fg_stt(19),'numfibers',300,'color',[.1 .3 1],'radius',.3,'newfig',0)
AFQ_RenderFibers(fg_stt(3),'numfibers',300,'color',[.1 .3 1],'radius',.3,'newfig',0)
axis image off
print('-dpng','-r200','~/arcuate+cst_stt+mrt_lat.png')
view(0,0);camlight(lh)
print('-dpng','-r200','~/arcuate+cst_stt+mrt_post.png')

cd ..
t1 = readFileNifti('t1/t1_class.nii.gz');
t1.data=t1.data==3;
msh = AFQ_meshCreate(t1,'smooth',300);
msh = AFQ_meshAddFgEndpoints(msh,fg_mrt(19), [1 .3 .1])
msh = AFQ_meshAddFgEndpoints(msh,fg_stt(19), [.1 .3 1])
AFQ_RenderCorticalSurface(msh)

msh = AFQ_meshCreate(t1)
[~,~,lh]=AFQ_RenderCorticalSurface(msh);hold
for ii=1:3:length(fg_mrt(19).fibers)
    AFQ_RenderEllipsoid(eye(3).*2,fg_mrt(19).fibers{ii}(:,end),10,[1 .3 .1],0);
    AFQ_RenderEllipsoid(eye(3).*2,fg_mrt(19).fibers{ii}(:,1),10,[1 .3 .1],0);
end
for ii=1:length(fg_stt(19).fibers)
    AFQ_RenderEllipsoid(eye(3).*2,fg_stt(19).fibers{ii}(:,end),10,[.1 .3 1],0);
    AFQ_RenderEllipsoid(eye(3).*2,fg_stt(19).fibers{ii}(:,1),10,[.1 .3 1],0);
end
view(250,0);camlight(lh,'right');axis off
print('-dpng','-r200','~/arcuate+cst_stt+mrt_surface.png')
