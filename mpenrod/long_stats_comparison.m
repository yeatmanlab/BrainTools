


inter_dir = '/mnt/scratch/projects/freesurfer/Intervention_Data';
control_dir = '/mnt/scratch/projects/freesurfer/Control_Data';

lh_inter_avg = importdata(fullfile(inter_dir,'consol_long.lh.aparc.stats.thickness-avg.dat'));
lh_inter_avg_labels = lh_inter_avg.textdata(1,2:36);
lh_inter_avg_data = lh_inter_avg.data(:,1:35);

lh_control_avg = importdata(fullfile(control_dir,'consol_long.lh.aparc.stats.thickness-avg.dat'));
lh_control_avg_labels = lh_control_avg.textdata(1,2:36);
lh_control_avg_data = lh_control_avg.data(:,1:35);

for ii = 1:cols(lh_inter_avg_data)
    lh_inter_avg_avgs(ii,1) = ii;
    lh_inter_avg_avgs(ii,2) = mean(lh_inter_avg_data(:,ii));
end
for ii = 1:cols(lh_control_avg_data)
    lh_control_avg_avgs(ii,1) = ii;
    lh_control_avg_avgs(ii,2) = mean(lh_control_avg_data(:,ii));
end
figure
hold on
scatter((1:rows(lh_inter_avg_data)),lh_inter_avg_data(:,2),'r')
scatter((1:rows(lh_control_avg_data)),lh_control_avg_data(:,2),'g')
title(lh_inter_avg_labels(1))
hold off
%% RH
rh_inter_avg = importdata(fullfile(inter_dir,'consol_long.rh.aparc.stats.thickness-avg.dat'));
rh_inter_avg_labels = rh_inter_avg.textdata(1,2:36);
rh_inter_avg_data = rh_inter_avg.data(:,1:35);

rh_control_avg = importdata(fullfile(control_dir,'consol_long.rh.aparc.stats.thickness-avg.dat'));
rh_control_avg_labels = rh_control_avg.textdata(1,2:36);
rh_control_avg_data = rh_control_avg.data(:,1:35);

for ii = 1:cols(rh_inter_avg_data)
    rh_inter_avg_avgs(ii,1) = ii;
    rh_inter_avg_avgs(ii,2) = mean(rh_inter_avg_data(:,ii));
end
for ii = 1:cols(rh_control_avg_data)
    rh_control_avg_avgs(ii,1) = ii;
    rh_control_avg_avgs(ii,2) = mean(rh_control_avg_data(:,ii));
end
figure
hold on
scatter((1:rows(rh_inter_avg_data)),rh_inter_avg_data(:,1),'r')
scatter((1:rows(rh_control_avg_avgs)),rh_control_avg_data(:,1),'g')
title(rh_inter_avg_labels(1))
hold off