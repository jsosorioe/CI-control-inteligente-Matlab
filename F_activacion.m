function [fun,der] = activation(s,alpha)

fun = 1./(1+exp(-alpha*s));
der = fun*(1-fun);

end 