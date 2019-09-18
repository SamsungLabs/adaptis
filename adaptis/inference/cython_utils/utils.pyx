# distutils: language = c++

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.queue cimport queue
from libcpp.pair cimport pair

cdef extern from "<utility>" namespace "std" nogil:
    pair[T,U] make_pair[T,U](T&,U&)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void bfs_cc(int x, int y, int color, float eps,
                 np.ndarray[np.float32_t, ndim=2, mode="c"] prob_map,
                 np.ndarray[np.int32_t, ndim=2, mode="c"] colors):
    cdef int *dxy = [-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1]
    cdef int k, nx, ny
    cdef float v_prob, u_prob
    cdef queue[pair[int, int]] q

    colors[x, y] = color

    cdef int h = prob_map.shape[0]
    cdef int w = prob_map.shape[1]

    q.push(make_pair(<int>x, <int>y))
    while not q.empty():
        v = q.front()
        q.pop()
        x, y = v.first, v.second
        v_prob = prob_map[x, y]

        for k in range(8):
            nx = x + dxy[2 * k]
            ny = y + dxy[2 * k + 1]
            if nx < 0 or nx >= h or ny < 0 or ny >= w:
                continue

            if colors[nx, ny] != -1:
                continue

            u_prob = prob_map[nx, ny]
            if v_prob + eps > u_prob:
                q.push(make_pair(<int>nx, <int>ny))
                colors[nx, ny] = color



def find_local_maxima(np.ndarray[np.float32_t, ndim=2, mode="c"] prob_map,
                      float prob_thresh=0.05, float eps=1e-3, float step_size=0.025):
    cdef int *dxy = [-1, -1, -1, 0, -1, 1, 0, -1, 0, 1, 1, -1, 1, 0, 1, 1]
    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] colors
    cdef int h = prob_map.shape[0]
    cdef int w = prob_map.shape[1]
    cdef int i, j, k, nx, ny, is_good_point
    cdef float v_prob, u_prob, prob
    cdef int last_color = 0
    cdef vector[pair[int, int]] rpoints

    colors = np.full((h, w), fill_value=-1, dtype=np.int32, order="C")
    for prob from 0.99 >= prob > prob_thresh by step_size:
        for i in range(h):
            for j in range(w):
                if colors[i, j] != -1:
                    continue

                is_good_point = 1
                v_prob = prob_map[i, j]
                if v_prob < prob:
                    continue

                for k in range(8):
                    nx = i + dxy[2 * k]
                    ny = j + dxy[2 * k + 1]
                    if nx < 0 or nx >= h or ny < 0 or ny >= w:
                        continue

                    u_prob = prob_map[nx, ny]
                    if u_prob < prob_thresh or v_prob < u_prob:
                        is_good_point = 0
                        break

                if is_good_point == 1:
                    rpoints.push_back(make_pair(<int>i, <int>j))
                    bfs_cc(i, j, last_color, eps, prob_map, colors)
                    last_color += 1

    return colors, rpoints
