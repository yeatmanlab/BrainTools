function R = fsl_rotMatrixFromEddy(x,y,z)
% Build a rotation matrix from the output of fsl's eddy
%
% R = fsl_rotMatrixFromEddy(x,y,z)
%
% this function returns R for rotation equation: x' = Rx
% R = inv(RxRyRz)

Rx = [1 0 0; 0 cos(x) sin(x);0 -sin(x) cos(x)];
Ry = [cos(y) 0 -sin(y); 0 1 0;sin(y) 0 cos(y)];
Rz = [cos(z) sin(z) 0; -sin(z) cos(z) 0;0 0 1];

R = inv(Rx*Ry*Rz);