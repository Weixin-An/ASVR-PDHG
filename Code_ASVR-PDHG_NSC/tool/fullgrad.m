function gold=fullgrad(zz, yy, type)
zy    = zz.*yy;
tempy = exp(-zy);
tempy = tempy./(1+tempy);
tempy = -yy.*tempy;
gold  = tempy;
end