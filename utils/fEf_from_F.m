% Copyright (c) 2023 NEC Corporation
% Focal legnth extraction from a fundamental matrix
%
% USAGE:
%   [f, E] = fEf_from_F(F, c1, c2, s)
% INPUTS:
%   F - 3x3, fundamental matrix
%   c1,c2 - 1x2, image center of each camera (default: [0,0])
%   s - 1x1, scaling factor (default: 1)
% OUTPUTS:
%   f - 1x1, focal length
%   E - 3x3, essential matrix
%   f and E are empty if the focal length estimation fails.
function [f, E] = fEf_from_F(F, c1, c2, s)
arguments
    F  (3,3) double
    c1 (1,2) double = [0,0]
    c2 (1,2) double = [0,0]
    s  (1,1) double = 1
end

    T1 = [s 0 c1(1)
          0 s c1(2)
          0 0    1];
    T2 = [s 0 c2(1)
          0 s c2(2)
          0 0    1];
    F = T2' * F * T1;


    f11 = F(1,1); f12 = F(1,2); f13 = F(1,3);
    f21 = F(2,1); f22 = F(2,2); f23 = F(2,3); 
    f31 = F(3,1); f32 = F(3,2); f33 = F(3,3); 

    num = -f13^2*f32*f33 - f23^2*f32*f33 + f12*f13*f33^2 + f22*f23*f33^2;
    den = f11*f13*f31*f32 + f21*f23*f31*f32 + f12*f13*f32^2 + f22*f23*f32^2 - f11*f12*f31*f33 - f21*f22*f31*f33 - f12^2*f32*f33 - f22^2*f32*f33;
    f2  = num / den;

    
    if f2 < 0
        f = [];
        E = [];
    else
        f = s * sqrt(f2);
        F = T2' \ F / T1;
        K1 = [f 0 c1(1)
              0 f c1(2)
              0 0    1];
        K2 = [f 0 c2(1)
              0 f c2(2)
              0 0    1];

        E = K2' * F * K1;
        E = E / norm(E,'fro');
    end

end