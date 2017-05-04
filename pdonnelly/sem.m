%% Function to calculate SEM
function [sem] = sem(data);

sem = nanstd(data)/sqrt(length(data));

return
