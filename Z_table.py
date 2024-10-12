from scipy.integrate import quad
from numpy import dot, divide, power, sqrt, e, inf, subtract, add

# Função de densidade de probabilidade normal Z~N(0, 1)
def PDF_norm(x):
    return dot(
        0.39894228040143267793994605993438,
        power( e, dot( -0.5, power( x, 2 ) ) )
    )

# Normaliza V.A. de X~N(u, o2) para Z~N(0, 1)
def Z_score(x, u, o2): return divide( subtract( x, u ), sqrt(o2) )

def VA(z, u, o2): return add( u, dot( z, sqrt(o2) ) )

# Função normal padrão acumulada à esquerda P(Z <= z) ou A(z)
def Z_left_norm(z): return -round(quad(PDF_norm, inf, -z)[0], 4)

# Função normal padrão acumulada à direita P(Z >= z)
def Z_right_norm(z): return round(quad(PDF_norm, z, inf)[0], 4)

# Forma alternativa das funções normal padrão acumulada invertendo a aréa
def Z_left_norm2(z): return round(quad(PDF_norm, -z, inf)[0], 4)
def Z_right_norm2(z): return -round(quad(PDF_norm, inf, z)[0], 4)

# Forma alternativa das funções normal padrão acumulada usando a formula inversa
def Z_left_norm3(z): return 1-Z_right_norm(z)
def Z_right_norm3(z): return 1-Z_left_norm(z)

# atalho para as funções normal padrão acumulada
def Zl(z): return Z_left_norm(z)
def Zr(z): return Z_right_norm(z)

# atalho para a normalização da V.A.
def Z(x, u, o2): return Z_score(x, u, o2)

# atalho para reversão da normalização
def X(z, u, o2): return VA(z, u , o2)

# atalhos para a função normal padrão acumulada com V.A. não normalizada
def Al(x, u, o2): return round(Zl(Z(x, u, o2)), 4)
def Ar(x, u, o2): return round(Zr(Z(x, u, o2)), 4)

# desenha a tabela para a função normal padrão acumulada
def print_Z_table(Z_norm):
    print("   ", end="")
    for Sz in range(0, 10):
        print("     {}".format(Sz), end="")
    print()

    print("      ", end="")
    for Sz in range(0, 10):
        print("________", end="")
    print()

    for Pz in range(-40, 41):
        Pz = Pz/10
        print("{: .1f}  |".format(Pz), end="")
        for Sz in range(0, 10):
            Sz = Sz/100
            z = Pz+(-Sz if Pz<0 else Sz)
            print("{: .4f}".format(Z_norm(z)), end=" ")
        print()

