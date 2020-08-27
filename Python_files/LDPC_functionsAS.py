import numpy as np
from math import exp, log
from numpy.random import randint


def quasi_cyclic() :
    I = np.eye(27)
    O = np.zeros((27, 27))
    Is = [np.roll(I, -p, 0) for p in range(27)]
    H = np.block([[Is[0], O, O, O, Is[0], Is[0], O, O, Is[0], O, O, Is[0], Is[1], Is[0], O, O, O, O, O, O, O, O, O, O], 
                 [Is[22], Is[0], O, O, Is[17], O, Is[0], Is[0], Is[12], O, O, O, O, Is[0], Is[0], O, O, O, O, O, O, O, O, O],
                 [Is[6], O, Is[0], O, Is[10], O, O, O, Is[24], O, Is[0], O, O, O, Is[0], Is[0], O, O, O, O, O, O, O, O],
                 [Is[2], O, O, Is[0], Is[20], O, O ,O, Is[25], Is[0], O, O, O, O, O, Is[0], Is[0], O, O, O, O, O, O, O],
                 [Is[23], O, O, O, Is[3], O, O, O, Is[0], O, Is[7], Is[11], O, O, O, O, Is[0], Is[0], O, O, O, O, O, O],
                 [Is[24], O, Is[23], Is[1], Is[17], O, Is[3], O, Is[10], O, O, O, O, O, O, O, O, Is[0], Is[0], O, O, O, O, O],
                 [Is[25], O, O, O, Is[8], O, O, O, Is[7], Is[18], O, O, O, O, O, O, O, O, Is[0], Is[0], O, O, O, O],
                 [Is[13], Is[24], O, Is[16], Is[0], O, Is[8], O, Is[6], O, O, O, O, O, O, O, O, O, O, Is[0], Is[0], O, O, O],
                 [Is[7], Is[20], O, O, Is[22], Is[10], O, O, Is[23], O, O, O, O, O, O, O, O, O, O, O, Is[0], Is[0], O, O],
                 [Is[11], O, O, O, Is[19], O, O, O, Is[13], O, Is[3], Is[17], O, O, O, O, O, O, O, O, O, Is[0], Is[0], O],
                 [Is[25], O, Is[8], O, Is[23], Is[18], O, Is[14], Is[9], O, O, O, O, O, O, O, O, O, O, O, O, O, Is[0], Is[0]],
                 [Is[3], O, O, O, Is[16], O, O, Is[2], Is[25], Is[5], O, O, O, O, O, O, O, O, O, O, O, O, O, Is[0]]])
    return H


def build_matrix(m, n) :
    cols = randint(3, 8, n)
    rows = randint(4, 11, m)
    cols_cnt = [0 for _ in range(n)]
    H = np.zeros((m, n), dtype = int)
    
    for i in range(m) :
        indices = randint(0, n, rows[i])
        for index in indices :
            j = index
            while (cols_cnt[index] > cols[index] and j < n - 1) :
                j = j + 1
            cols_cnt[j] += 1
            H[i, j] = 1
    return H


def parity_matrix(n, p) :
    if n == 648 :
        H = quasi_cyclic()
    else :
        if p <= 0.02 :
            H = build_matrix(n // 5, n)
        elif p <= 0.04 :
            H = build_matrix(n // 3, n)
        elif p <= 0.08 :
            H = build_matrix(n // 2, n)
        else :
            H = build_matrix(2 * n // 3, n)
    return H


def lookup_tables(H) :
    LL_index = [0]
    LL_groups = 1
    for row in np.transpose(H) :
        LL_index.append(LL_groups)
        LL_groups += sum(row)
    LL_index.append(len(LL_index))
    
    cs_index = [0]
    cs_groups = 1
    for row in H :
        cs_index.append(cs_groups)
        cs_groups += sum(row)
    cs_index.append(len(cs_index))
    
    cs_list = [0]
    n = len(H[0])
    m = len(H)
    bit_cs = [0 for _ in range(n + 1)]
    for row in range(m) :
        for bit in range(n) :
            if H[row, bit] == 1 :
                cs_list.append([bit + 1, LL_index[bit + 1] + bit_cs[bit + 1]])
                bit_cs[bit + 1] += 1
                
    return np.array(LL_index), np.array(cs_index), np.array(cs_list)


def non_zeros(M) :
    return sum([sum(row) for row in M])


def syndrome(M1, x) :
    return np.matmul(M1, np.transpose(np.array(x)))


def produce_a2f(p, Na, Nf, Ma, Mf) :
    a2f = [0]
    for i in range(1, Ma + 1) :
        a = exp(-i/Na)
        f = (1 + a)/(1 - a)
        z = log(f)
        j = int(Nf * z + 0.5) 
        if j <= 0 :
            j = 1
        if j > Mf :
            j = Mf
        a2f.append(j)
    return a2f


def produce_f2a(p, Na, Nf, Ma, Mf) :
    f2a = [0]
    for i in range(1, Mf + 1) :
        f = exp(i/Nf)
        a = (f - 1)/(f + 1)
        z = -log(a)
        j = int(Na * z + 0.5) 
        if j <= 0 :
            j = 1
        if j > Ma :
            j = Ma
        f2a.append(j)
    return f2a


def LL_init1(Na, Nf, k, p) :
    LL_reg = [0]
    f_init = Nf * log((1-p)/p)
    a_init = Na * log(1 - 2 * p)
    for i in range(1, k + 1) :
        LL_reg.append(a_init)
    return f_init, LL_reg


def cs_msgs_2_bits(m, d, cs_index, cs_list, LL_reg, Ma) : 
    for i in range(1, m + 1) :
        sign = d[i]
        big_alpha = 0
        j1 = cs_index[i]
        j2 = cs_index[i + 1]
        for j in range(j1, j2) :
            a1 = LL_reg[cs_list[j][1]]
            if a1 < 0 :
                sign = 1 - sign
                big_alpha -= a1
        for j in range(j1, j2) :
            a1 = LL_reg[cs_list[j][1]]
            p_sign = sign
            if a1 < 0 :
                p_alpha = big_alpha + a1
            else :
                p_alpha = big_alpha + a1
            if p_alpha <= 0 :
                p_alpha = 1
            if p_alpha > Ma :
                p_alpha = Ma
            if p_sign == 0 :
                LL_reg[cs_list[j][1]] = p_alpha
            else :
                LL_reg[cs_list[j][1]] = - p_alpha
    return LL_reg


def bit_msgs_2_cs(n, f_init, a2f, Mf, LL_reg, y) :
    y1 = []
    for i in range(1, n + 1) :
        j1, j2 = LL_index[i], LL_index[i + 1]
        f_tot = f_init
        for j in range(j1, j2) :
            u = LL_reg[j]
            if u > 0 :
                u = a2f[u]
            else :
                u = - a2f[-u]
            LL_reg[j] = u
            f_tot += u
        
        for j in range(j1, j2) :
            x = f_tot - LL_reg[j]
            if x < 0 :
                p_sign = 1
                x = -x
            else :
                p_sign = 0
            if x < 1 :
                x = 1
            if x > Mf :
                x = Mf
            if p_sign == 1 :
                LL_reg[j] = - f2a[x]
            else :
                LL_reg[j] = f2a[x]
    
        if f_tot < 0 :
            y1.append(1 - y[i])
        else : 
            y1.append(y[i])
        return y1, LL_reg


def converged(m, cs_index, cs_list, C) :
    success = 1
    for i in range(1, m + 1) :
        chksum = 0
        j1, j2 = cs_index[i], cs_index[i + 1]
        for j in range(j1, j2) :
            chksum = chksum ^ cs_list[j][0]
        if chksum != c[i] :
            success = 0
    return success


def belief_prop(C, D, MAX_ITERS, Na, Nf, Ma, Mf, p, k, y, H) :
    a2f, f2a = produce_a2f(p, Na, Nf, Ma, Mf), produce_f2a(p, Na, Nf, Ma, Mf)
    f_init, LL_reg = LL_init(Na, Nf, k, p)
    LL_index, cs_index, cs_list = lookup_tables(H)
    success = 0
    i = 0
    while (i < MAX_ITERS) :
        LL_reg = cs_msgs_2_bits(m, d, cs_index, cs_list, LL_reg, Ma)
        y, LL_reg = bit_msgs_2_cs(n, f_init, a2f, Mf, LL_reg, y)
        success = converged(m, cs_index, cs_list, C)
        i += 1
        if success == 1 : 
            break
    return y, success
