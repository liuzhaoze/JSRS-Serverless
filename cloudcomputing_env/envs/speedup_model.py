def SU(A, sigma, n):
    if 0 <= sigma <= 1:
        if 1 <= n <= A:
            return A * n / (A + sigma * (n - 1) / 2)
        elif n <= 2 * A - 1:
            return A * n / (sigma * (A - 0.5) + n * (1 - sigma / 2))
        else:
            return A
    else:
        if 1 <= n <= A + A * sigma - sigma:
            return n * A * (sigma + 1) / (A + A * sigma - sigma + n * sigma)
        else:
            return A
