import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t Float
ctypedef np.int64_t Long

@cython.boundscheck(False)
@cython.wraparound(False)

def compute_mat(np.ndarray[Float,ndim=2] mat,np.ndarray[Float,ndim=2] rois,np.ndarray[Float,ndim=1] score):
    cdef int x,y,r,start_y,end_y,start_x,end_x
    for r in range(len(rois)):
        start_y = int(rois[r,1])
        end_y = int(rois[r,3])
        start_x = int(rois[r,0])
        end_x = int(rois[r,2])
        for y in range(start_y,end_y):
            for x in range(start_x,end_x):
                mat[y,x] += score[r]
    return mat

def icr(np.ndarray[Float,ndim=2] mat,np.ndarray[Float,ndim=3] y_k,np.ndarray[Float,ndim=2] w,np.ndarray[Float,ndim=2] I,np.ndarray[Float,ndim=1] x_list,np.ndarray[Long,ndim=1] j_list,float I_t,int c,int batch):
    cdef int i,t,a,b
    a = len(w[batch])
    b = len(j_list)
    for i in range(b):
        for t in range(a):
            if mat[t,i]>I[batch,t]:
                w[batch,t] = x_list[i]
                I[batch,t] = mat[t,i]
            if mat[t,i]>I_t:
                y_k[batch,t,3] = 0
                y_k[batch,t,c] = 1
    return y_k, w, I