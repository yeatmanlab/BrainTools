function frameorder = makeFrameOrder(cond, repim, repb)

% Loop over blocks which are rows of cond
for ii = 1:size(cond,1)
    trial = [];
    % Loop over trials which are columns of cond
    for jj = 1:size(cond,2)
        trial = horzcat(trial,repmat(cond(ii,jj),[1 repim]), zeros(1,repb));
    end
    frameorder(ii,:) = trial;
end