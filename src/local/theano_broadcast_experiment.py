import theano
import theano.tensor as T
from theano import pp


s1 = T.iscalar("s1")
v1 = T.ivector("v1")
v2 = T.ivector("v2")
m1 = T.imatrix("m1")
m2 = T.imatrix("m2")

out1 = T.sum(m1[v1, v2])
f1 = theano.function([m1, v1, v2], out1)
print(theano.pp(out1))
m = [[1,2],[3,4]]
v = [0,1]
ret1 = f1(m, v, v)
print(ret1)

out2 = T.sum(m1*m2)
f2 = theano.function([m1, m2], out2)
print(theano.pp(out2))
id = [[0,1],[1,0]]
ret2 = f2(m, id)
print(ret2)

assert(ret1 == ret2)

out3 = T.sum(m1[v1, m2])
f3 = theano.function([m1, v1, m2], out3)
print(theano.pp(out3))
ret3 = f3(m, v, id)
print(ret3)