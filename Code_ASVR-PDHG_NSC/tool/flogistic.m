function ff = flogistic(zz, yy)
zy    = zz.*yy;
tempy = exp(-zy);
tempy = log(1+tempy);
tempy = sum(tempy);
ff    = tempy;
end
