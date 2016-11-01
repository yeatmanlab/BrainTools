function [s] = ifsig(x,y)

[r, p] = corr(x,y, 'rows', 'complete');
if p < 0.01
    s =  'r*';
elseif p < 0.05
    s =  'b*';
elseif p > 0.05
    s = 'k*';
end
