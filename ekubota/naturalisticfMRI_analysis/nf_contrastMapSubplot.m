
subs = {'PreK_PD','PreK_EK','PreK_EO','PreK_LG'};

figure;hold on;
for ii = 1:length(subs)
    dataPath = strcat('/mnt/scratch/PREK_Analysis/',subs{ii},'/nf_anatomy');
    cd(dataPath)
    load msh.mat
    im = 'AllvBaseline.nii.gz';
    msh2 = AFQ_meshColor(msh, 'overlay',im, 'crange',[-.4 .4]); %,'thresh', .2);
    colormap('jet')
    caxis([-0.4 0.4])
    subplot(2,2,ii)
    [p,~,L]=AFQ_RenderCorticalSurface(msh2); % Render cortex
    view(180,-90); % Camera angle, ventral surface.
    %view(-90,0);
    camlight(L,'left'); % move the light. you can also give azymuth and elevation
    axis off % turn of plot axis
    title(subs{ii});
    colorbar
end 