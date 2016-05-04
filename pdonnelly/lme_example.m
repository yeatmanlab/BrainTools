sIds  = [1 1 1 2 2 2 3 3]';
hours = [0 40 80 0 40 100 0 100]';
% Center (demean) hours
hours = hours - mean(hours);
hours2 = hours.^2;
BR    = [70 75 80 90 95 100 100 105]';
gender= [1 1 1 0 0 0 1 1];
% Create a matlab "dataset" variable
DS = dataset(sIds,hours,hours2,BR);





% Make sIds a categorical variable
DS.sIds = nominal(DS.sIds);
% Fit the model where we predict BR as changing linearly with the number of
% hours of intervention
lme = fitlme(DS, 'BR ~ hours + (1|sIds)');
% Fit the model where we predict BR as changing quadratically with hours of
% intervention
lme2 = fitlme(DS, 'BR ~ hours + hours2 + (1|sIds)');
