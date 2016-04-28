function a = testingPar(A)

tic
parfor i = 1:200
    a(i) = max(abs(eig(rand(A))));
end
toc

