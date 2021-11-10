% calculate subgradient for hinge loss
function g = sub_grad(x, sample, label)
    d = size(sample,1);
    if (label*(x'*sample)) < 1.0
        g = -label*sample;
    else
        g = zeros(d,1);
    end
end