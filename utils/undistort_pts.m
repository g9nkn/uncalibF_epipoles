% Copyright (c) 2023 NEC Corporation
% Undistort image points based on the division model (A. Fitzgibbon)
%
% USAGE:
%   mu = undistort_pts(md, k, c)
% INPUTS:
%   md - 2xN, distorted points
%   k  - 1x1 or 1x2 or 1x3, radial distortion parameters
%   c  - 1x2, image center
% OUTPUTS:
%   mu - 2xN, undistorted points
function mu = undistort_pts(md, k, c)
    if nargin < 3, c = zeros(2,1); end

    xd = md(1,:) - c(1);
    yd = md(2,:) - c(2);
    
    % radial distortion term
    r2 = xd.^2 + yd.^2;
    r4 = r2.^2;
    r6 = r2.*r4;
    switch length(k)
        case 1
            radial = 1 + k(1)*r2;
        case 2
            radial = 1 + k(1)*r2 + k(2)*r4;
        case 3
            radial = 1 + k(1)*r2 + k(2)*r4 + k(3)*r6;
    end
    
    % undistorted coordinates (shift the origin to [0,0])
    xu = xd ./ radial;
    yu = yd ./ radial;

    mu = [xu + c(1);
          yu + c(2)];

    if size(md,1) == 3
        mu(3,:) = 1;
    end
    
end