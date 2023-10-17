% Copyright (c) 2023 NEC Corporation
function S = skew3x3(a)
    S = [0, -a(3), a(2)
         a(3), 0, -a(1)
         -a(2), a(1), 0];
end