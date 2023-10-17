% Ito's 3-point method for fundamental matrix estimation with two known epipoles
%
% USAGE:
%   [F, f, E] = F3pt_e1e2_Ito(e1, e2, m1, m2, c1, c2)
% INPUTS:
%   e1,e2 - 3x1, epipoles in homogeneous coordinates
%   m1,m2 - 3xN, (N>=3) 2D point correspondences in homogeneous coordinates 
%   c1,c2 - 1x2, image center of each camera (default: [0,0])
% OUTPUTS:
%   F - 3x3, fundamental matrix
%   f - 1x1, focal length
%   E - 3x3, essential matrix
% REFERENCE:
%   M. Ito, T. Sugimura, and J. Sato. Recovering structures and 
%   motions from mutual projection of cameras. ICPR2002.
function [F, f, E] = F3pt_e1e2_Ito(e1, e2, m1, m2, c1, c2)
arguments 
    e1 (3,1) double
    e2 (3,1) double
    m1 (3,:) double
    m2 (3,:) double
    c1 (1,2) double = [0,0]
    c2 (1,2) double = [0,0]
end

if ~(all(size(m1)==size(m2)) && size(m1,2)>=3)
    error("Requireds >=3 points.");
end

e1 = e1 / e1(3);
e2 = e2 / e2(3);
m1 = m1 ./ m1(3,:);
m2 = m2 ./ m2(3,:);

s  = max(abs([m1(:);m2(:);e1;e2]));
T1 = [1/s 0 -c1(1)/s
      0 1/s -c1(2)/s
      0   0      1];
T2 = [1/s 0 -c2(1)/s
      0 1/s -c2(2)/s
      0   0    1];
m1 = T1*m1;
m2 = T2*m2;
e1 = T1*e1;
e2 = T2*e2;

  
O13 = zeros(1,3);
A = [e1', O13, O13
     O13, e1', O13
     O13, O13, e1'
     e2(1)*eye(3), e2(2)*eye(3), e2(3)*eye(3)];
[~,~,V] = svd(A);
N = V(:,6:9);

B = [m2(1,:)'.*m1', m2(2,:)'.*m1', m2(3,:)'.*m1'] * N;
[~,~,V] = svd(B'*B);
f_vec = N * V(:,end);

F_tmp = reshape(f_vec,3,3)';
[U,D,V] = svd(F_tmp);
F_tmp = U * diag([D(1,1),D(2,2),0]) * V';
F_tmp = T2' * F_tmp * T1;
F     = F_tmp / norm(F_tmp,'fro');

if nargout > 1
    [f, ~] = fEf_from_F(F,c1,c2,s);

    if isempty(f)
        E = [];
        return
    end
    K1 = [f  0 c1(1)
          0  f c1(2)
          0  0   1];
    K2 = [f  0 c2(1)
          0  f c2(2)
          0  0   1];
    
    E_tmp = K2' * F * K1;
    E = E_tmp / norm(E_tmp,'fro');
end        

end
